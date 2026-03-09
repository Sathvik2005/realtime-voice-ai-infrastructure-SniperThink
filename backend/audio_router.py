"""
Audio Router  —  Pipeline Orchestrator

Central coordinator for one voice session:

  Browser mic audio
    ↓  (WebRTC → 16 kHz int16 PCM)
  process_audio_chunk()
    ↓
  VAD  ──────────────────────────────────┐
    speech_start → LISTENING             │ interrupt?
    speech_end   → PROCESSING  ─────────┼──────────────────────────┐
    ↓                                    │                          │
  Faster-Whisper  (ASR)                  │                          │
    ↓  transcript                        │                          │
  Optional RAG retrieval                 │                          ▼
    ↓  context                           │             InterruptController.interrupt()
  LLM (streaming) ──────── text tokens ──┴─► TTS (streaming) ──► WebSocket binary
    ↓  full response
  Update conversation history

Module-level singletons for heavy models (VAD model tensor, Whisper model,
OpenAI client, TTS engine) are loaded once at startup and shared read-only
across all sessions.  Each session has its own VADEngine and StreamingASR
instance so per-session mutable state remains isolated.
"""

import asyncio
import json
import logging
import struct
import os
from typing import Optional

from session_manager import Session, SessionState
from vad import VADEngine
from asr_streaming import StreamingASR
from llm_engine import LLMEngine
from tts_streaming import StreamingTTS
from interrupt_controller import InterruptController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level shared models  (initialised once at startup)
# ---------------------------------------------------------------------------
_llm: Optional[LLMEngine] = None
_tts: Optional[StreamingTTS] = None


async def initialize_models() -> None:
    """Load / initialise all ML models.  Called from FastAPI lifespan."""
    global _llm, _tts

    # 1. Silero VAD — requires torch; skip gracefully if unavailable
    try:
        await VADEngine.load_shared_model()
        logger.info("VAD model loaded.")
    except Exception as exc:
        logger.warning(f"VAD not available (torch missing?): {exc}")

    # 2. Faster-Whisper ASR — requires faster_whisper; skip gracefully
    try:
        whisper_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
        whisper_device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute_type = "int8" if whisper_device == "cpu" else "float16"
        await StreamingASR.load_shared_model(
            model_size=whisper_size,
            device=whisper_device,
            compute_type=compute_type,
        )
        logger.info(f"Whisper ASR loaded: {whisper_size} on {whisper_device}.")
    except Exception as exc:
        logger.warning(f"ASR not available (faster-whisper missing?): {exc}")

    # 3. LLM — always attempt; needs only openai + OPENAI_API_KEY in env
    try:
        _llm = LLMEngine(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
        await _llm.initialize()
        logger.info("LLM ready.")
    except Exception as exc:
        logger.error(f"LLM initialisation failed: {exc}")

    # 4. TTS — requires piper binary; skip gracefully
    try:
        _tts = StreamingTTS(voice=os.environ.get("TTS_VOICE", "en_US-lessac-medium"))
        logger.info("TTS ready.")
    except Exception as exc:
        logger.warning(f"TTS not available (piper missing?): {exc}")

    logger.info("Pipeline initialisation complete (demo-mode if ML libs absent).")


# ---------------------------------------------------------------------------
# AudioRouter
# ---------------------------------------------------------------------------
class AudioRouter:
    """
    One instance per session.  Owns per-session VAD state and ASR buffer.
    References shared LLM and TTS singletons.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.vad = VADEngine()          # per-session VAD state machine
        self.asr = StreamingASR()       # per-session audio buffer
        self.llm = _llm                 # shared, stateless (history passed in)
        self.tts = _tts                 # shared, stateless

        self.interrupt_ctrl = InterruptController(session)
        self._pipeline_task: Optional[asyncio.Task] = None

        # Wire VAD callbacks
        self.vad.on_speech_start = self._on_speech_start
        self.vad.on_speech_end = self._on_speech_end

    # ------------------------------------------------------------------
    # VAD loading (must be awaited after construction)
    # ------------------------------------------------------------------
    async def load_vad(self) -> None:
        """No-op: VAD model is loaded at the class level during startup."""
        pass  # VADEngine.load_shared_model() already called in initialize_models()

    # ------------------------------------------------------------------
    # Main entry point — called per WebRTC audio frame
    # ------------------------------------------------------------------
    async def process_audio_chunk(self, pcm_bytes: bytes) -> None:
        """
        Route every incoming 16 kHz int16 PCM chunk through VAD.
        Buffer audio to ASR while the user is speaking.
        Detect interruptions when the user speaks over the AI.
        """
        events = await self.vad.process_chunk(pcm_bytes)

        for event in events:
            if event == "speech_start":
                if self.session.state == SessionState.SPEAKING:
                    # User interrupted the AI — cancel immediately
                    asyncio.create_task(self.handle_interrupt())

        # Accumulate audio in ASR buffer while the user is actively speaking
        if self.vad._speech_active and self.session.state == SessionState.LISTENING:
            self.asr.append(pcm_bytes)

    # ------------------------------------------------------------------
    # VAD callbacks
    # ------------------------------------------------------------------
    async def _on_speech_start(self) -> None:
        logger.info(f"[{self.session.session_id}] Speech start")
        self.session.transition(SessionState.LISTENING)
        self.asr.clear()
        await self._send_event("speech_start")

    async def _on_speech_end(self) -> None:
        logger.info(f"[{self.session.session_id}] Speech end")
        if self.session.state != SessionState.LISTENING:
            return

        self.session.transition(SessionState.PROCESSING)
        await self._send_event("speech_end")

        # Launch the processing pipeline without blocking the VAD loop
        self._pipeline_task = asyncio.create_task(
            self._run_pipeline(),
            name=f"pipeline-{self.session.session_id}",
        )
        self.interrupt_ctrl.set_pipeline_task(self._pipeline_task)

    # ------------------------------------------------------------------
    # Interrupt handling
    # ------------------------------------------------------------------
    async def handle_interrupt(self) -> None:
        """Cancel any running pipeline and return to IDLE."""
        await self.interrupt_ctrl.interrupt()
        self.vad.reset()
        self.session.transition(SessionState.IDLE)
        await self._send_event("interrupted")

    # ------------------------------------------------------------------
    # Processing pipeline  (ASR → LLM → TTS)
    # ------------------------------------------------------------------
    async def _run_pipeline(self) -> None:
        try:
            # ── Stage 1: Transcribe ───────────────────────────────────
            transcript = await self.asr.final()
            if not transcript:
                self.session.transition(SessionState.IDLE)
                return

            await self._send_event("transcript", {"text": transcript})
            logger.info(f"[{self.session.session_id}] Transcript: '{transcript}'")

            # ── Stage 2: Optional RAG retrieval ──────────────────────
            context = ""
            rag_enabled = os.environ.get("RAG_ENABLED", "false").lower() == "true"
            if rag_enabled:
                try:
                    from rag.retrieval import retrieve_context
                    context = await retrieve_context(transcript)
                except Exception as rag_err:
                    logger.warning(f"RAG retrieval skipped: {rag_err}")

            # ── Stage 3: LLM + TTS concurrently ──────────────────────
            self.session.transition(SessionState.SPEAKING)
            text_queue: asyncio.Queue = asyncio.Queue()
            full_response = ""

            async def llm_worker():
                nonlocal full_response
                async for token in self.llm.stream_response(
                    transcript,
                    self.session.conversation_history,
                    context,
                    self.session.interrupt_event,
                ):
                    full_response += token
                    await text_queue.put(token)
                await text_queue.put(None)  # End sentinel

            async def tts_worker():
                async def token_gen():
                    while True:
                        tok = await text_queue.get()
                        if tok is None:
                            return
                        yield tok

                async for audio_chunk in self.tts.synth_stream(
                    token_gen(), self.session.interrupt_event
                ):
                    await self._send_audio(audio_chunk)

            await asyncio.gather(llm_worker(), tts_worker())

            # ── Stage 4: Update conversation history ─────────────────
            if full_response:
                self.session.conversation_history = LLMEngine.update_history(
                    self.session.conversation_history, transcript, full_response
                )

            self.session.transition(SessionState.IDLE)
            await self._send_event("response_complete")

        except asyncio.CancelledError:
            logger.info(f"[{self.session.session_id}] Pipeline cancelled (interrupt)")
            self.session.transition(SessionState.IDLE)
            raise
        except Exception as exc:
            logger.error(
                f"[{self.session.session_id}] Pipeline error: {exc}", exc_info=True
            )
            self.session.transition(SessionState.IDLE)

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------
    async def _send_event(self, event_type: str, data: dict = None) -> None:
        """Send a JSON control event to the browser over WebSocket."""
        ws = self.session.websocket
        if ws is None:
            return
        msg = {"type": event_type}
        if data:
            msg.update(data)
        try:
            await ws.send_text(json.dumps(msg))
        except Exception as exc:
            logger.debug(f"send_event failed: {exc}")

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """
        Send a TTS audio chunk to the browser as a binary WebSocket frame.
        Frame layout: [0x01][PCM audio bytes]
          0x01 = audio payload type marker (reserved for future message types)
        """
        ws = self.session.websocket
        if ws is None:
            return
        try:
            header = struct.pack("!B", 0x01)
            await ws.send_bytes(header + audio_bytes)
        except Exception as exc:
            logger.debug(f"send_audio failed: {exc}")
