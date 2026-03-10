"""
Streaming Speech Recognition

Two provider modes, selected via ASR_MODE env var:

  ASR_MODE=local (default)
    Uses faster-whisper running locally (GPU/CPU).
    Class: StreamingASR

  ASR_MODE=hf
    Uses the openai/whisper Gradio Space on Hugging Face via gradio_client.
    No local model download required — audio is sent over HTTPS.
    Requires: pip install gradio_client
    Optional: set HF_TOKEN env var for private spaces / higher rate limits.
    Class: HFWhisperASR

Both classes expose the same interface:
  append(pcm_bytes)  — buffer raw 16 kHz int16 PCM
  clear()            — discard buffer
  final()            — accurate transcription, clears buffer
  partial()          — quick live transcription, does not clear

Factory function:
  make_asr(mode)     — returns the right instance for the current mode
"""

import asyncio
import io
import logging
import os
import tempfile
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2  # int16


class StreamingASR:
    """
    Per-instance audio buffer + shared faster-whisper model.
    """

    _model = None  # Shared WhisperModel loaded once at startup

    # ------------------------------------------------------------------
    # Class-level model loading
    # ------------------------------------------------------------------
    @classmethod
    async def load_shared_model(
        cls,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        if cls._model is not None:
            return
        loop = asyncio.get_event_loop()
        cls._model = await loop.run_in_executor(
            None, cls._load_whisper, model_size, device, compute_type
        )
        logger.info(f"Faster-Whisper loaded: size={model_size} device={device}")

    @staticmethod
    def _load_whisper(model_size, device, compute_type):
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    # ------------------------------------------------------------------
    # Instance
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._buffer = bytearray()

    def append(self, pcm_bytes: bytes) -> None:
        """Append raw 16 kHz int16 PCM bytes to the utterance buffer."""
        self._buffer.extend(pcm_bytes)

    def clear(self) -> None:
        self._buffer = bytearray()

    def buffer_duration_ms(self) -> float:
        samples = len(self._buffer) // BYTES_PER_SAMPLE
        return samples / SAMPLE_RATE * 1_000

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    async def partial(self) -> str:
        """
        Quick transcription of current buffer for live display.
        Uses beam_size=1 for speed; accuracy is secondary here.
        """
        if self.buffer_duration_ms() < 500:
            return ""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe, 1)

    async def final(self) -> str:
        """
        Accurate transcription of the full utterance.
        Clears the buffer afterwards.
        """
        if self.buffer_duration_ms() < 200:
            self.clear()
            return ""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._transcribe, 5)
        logger.info(f"ASR final: '{result}'")
        self.clear()
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _transcribe(self, beam_size: int) -> str:
        if self._model is None:
            return ""

        audio = (
            np.frombuffer(bytes(self._buffer), dtype=np.int16).astype(np.float32)
            / 32_768.0
        )

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=beam_size,
            vad_filter=False,   # External VAD already handled
            word_timestamps=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# HF Whisper ASR  —  Hugging Face Gradio Space (openai/whisper)
# ---------------------------------------------------------------------------

class HFWhisperASR:
    """
    ASR provider that calls the openai/whisper Gradio Space on Hugging Face.
    Compatible interface with StreamingASR: append / clear / partial / final.

    Audio is buffered locally as raw PCM; on final() it is encoded to a
    temporary WAV file, submitted to the Space via gradio_client, and the
    returned transcript is returned.

    partial() returns a lightweight browser-SpeechRecognition-style hint
    (empty string while audio is still being collected) — real partial decodes
    aren't possible with a request/response API, so the live bubble is filled
    by the WebSocket transcript_direct path instead.
    """

    _client = None   # shared gradio_client.Client created once at startup

    @classmethod
    async def load_shared_client(
        cls,
        hf_token: Optional[str] = None,
        space: str = "openai/whisper",
    ) -> None:
        if cls._client is not None:
            return
        loop = asyncio.get_event_loop()
        cls._client = await loop.run_in_executor(
            None, cls._build_client, space, hf_token
        )
        logger.info(f"HF Whisper client ready: space={space}")

    @staticmethod
    def _build_client(space: str, hf_token: Optional[str]):
        from gradio_client import Client
        kwargs = {"hf_token": hf_token} if hf_token else {}
        return Client(space, **kwargs)

    # ------------------------------------------------------------------
    # Instance
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._buffer = bytearray()

    def append(self, pcm_bytes: bytes) -> None:
        self._buffer.extend(pcm_bytes)

    def clear(self) -> None:
        self._buffer = bytearray()

    def buffer_duration_ms(self) -> float:
        samples = len(self._buffer) // BYTES_PER_SAMPLE
        return samples / SAMPLE_RATE * 1_000

    async def partial(self) -> str:
        """
        HF Space is request/response — no streaming partial decodes.
        Return empty string; the browser's SpeechRecognition fills the live bubble.
        """
        return ""

    async def final(self) -> str:
        """
        Send the buffered audio to openai/whisper on HF Spaces and return
        the transcription.  The buffer is cleared afterwards.
        """
        if self.buffer_duration_ms() < 200:
            self.clear()
            return ""

        pcm_data = bytes(self._buffer)
        self.clear()

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._call_hf, pcm_data)
            logger.info(f"HF Whisper final: '{result}'")
            return result
        except Exception as exc:
            logger.warning(f"HF Whisper request failed: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _call_hf(self, pcm_data: bytes) -> str:
        """
        Write PCM to a temp WAV, submit to the Gradio Space, return text.
        Runs in a thread pool executor (blocking I/O).
        """
        if self._client is None:
            return ""

        # Encode raw int16 PCM → WAV bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
        wav_bytes = wav_buffer.getvalue()

        # gradio_client needs a file path — write to a named temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            result = self._client.predict(
                inputs=tmp_path,
                task="transcribe",
                api_name="/predict",
            )
            # result may be a dict {"text": "..."} or a plain string
            if isinstance(result, dict):
                return result.get("text", "").strip()
            return str(result).strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_asr(mode: str = "local") -> "StreamingASR | HFWhisperASR":
    """Return a fresh ASR instance for the given mode."""
    if mode == "hf":
        return HFWhisperASR()
    return StreamingASR()
