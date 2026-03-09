"""
WebSocket Signaling Handler

Routes JSON control messages between the browser and the backend:
  • offer / answer   — WebRTC SDP exchange
  • ice_candidate    — ICE connectivity checks
  • interrupt        — user manually stops AI speech
  • audio_data       — raw PCM fallback (when WebRTC is unavailable)

TTS audio chunks are sent back to the browser as binary WebSocket frames
with a 1-byte header (0x01 = audio payload).
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket

from session_manager import Session, SessionState
from webrtc_server import WebRTCServer

logger = logging.getLogger(__name__)


class SignalingHandler:
    def __init__(self, session: Session, websocket: WebSocket) -> None:
        self.session = session
        self.websocket = websocket
        self.webrtc = WebRTCServer(session)

        # Attach websocket to session so AudioRouter can push events/audio
        self.session.websocket = websocket

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Read messages until the connection closes."""
        await self.websocket.send_json(
            {"type": "ready", "session_id": self.session.session_id}
        )
        async for message in self._messages():
            await self._dispatch(message)

    # ------------------------------------------------------------------
    # Message dispatching
    # ------------------------------------------------------------------
    async def _dispatch(self, message: dict) -> None:
        handlers = {
            "offer": self._handle_offer,
            "ice_candidate": self._handle_ice_candidate,
            "interrupt": self._handle_interrupt,
            "audio_data": self._handle_direct_audio,
            # Demo mode: browser sends recognised text, backend runs LLM
            "transcript_direct": self._handle_transcript_direct,
        }
        handler = handlers.get(message.get("type"))
        if handler:
            await handler(message)
        else:
            logger.debug(f"[{self.session.session_id}] Unknown message type: {message.get('type')}")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    async def _handle_offer(self, message: dict) -> None:
        """Process WebRTC SDP offer; reply with answer."""
        try:
            answer = await self.webrtc.handle_offer(
                sdp=message["sdp"],
                type_=message.get("type_", "offer"),
            )
            await self.websocket.send_json(
                {"type": "answer", "sdp": answer["sdp"], "type_": answer["type"]}
            )
            logger.info(f"[{self.session.session_id}] WebRTC SDP handshake complete")
        except Exception as exc:
            logger.error(f"[{self.session.session_id}] Offer error: {exc}")
            await self.websocket.send_json({"type": "error", "message": str(exc)})

    async def _handle_ice_candidate(self, message: dict) -> None:
        """Forward browser ICE candidate to aiortc peer connection."""
        try:
            await self.webrtc.add_ice_candidate(message["candidate"])
        except Exception as exc:
            logger.warning(f"[{self.session.session_id}] ICE candidate error: {exc}")

    async def _handle_interrupt(self, _message: dict) -> None:
        """Browser requested an immediate interruption."""
        logger.info(f"[{self.session.session_id}] Client-triggered interrupt")
        self.session.interrupt_event.set()
        if self.webrtc.router:
            await self.webrtc.router.handle_interrupt()

    async def _handle_transcript_direct(self, message: dict) -> None:
        """
        Demo mode: browser sends a recognised-speech transcript as plain text.
        Backend runs LLM → streams response text back as JSON.
        No ASR / VAD / WebRTC needed — works with just an OpenAI API key.
        """
        transcript = message.get("text", "").strip()
        if not transcript:
            return

        logger.info(f"[{self.session.session_id}] transcript_direct: '{transcript}'")
        self.session.interrupt_event.clear()
        self.session.transition(SessionState.PROCESSING)

        # Echo transcript back for display
        await self.websocket.send_json({"type": "transcript", "text": transcript})

        asyncio.create_task(
            self._llm_respond(transcript),
            name=f"llm-{self.session.session_id}",
        )

    async def _llm_respond(self, transcript: str) -> None:
        """Run LLM and stream the text response back to the browser."""
        from audio_router import _llm
        from llm_engine import LLMEngine

        if _llm is None:
            await self.websocket.send_json({
                "type": "error",
                "message": "LLM not ready. Add OPENAI_API_KEY to .env and restart.",
            })
            self.session.transition(SessionState.IDLE)
            return

        self.session.transition(SessionState.SPEAKING)
        full_response = ""

        try:
            async for token in _llm.stream_response(
                transcript,
                self.session.conversation_history,
                "",
                self.session.interrupt_event,
            ):
                if self.session.interrupt_event.is_set():
                    break
                full_response += token
                # Stream tokens so UI can show partial text
                await self.websocket.send_json({"type": "response_token", "token": token})

            await self.websocket.send_json({
                "type": "llm_response",
                "text": full_response,
            })

            self.session.conversation_history = LLMEngine.update_history(
                self.session.conversation_history, transcript, full_response
            )
        except Exception as exc:
            logger.error(f"[{self.session.session_id}] LLM error: {exc}")
            # Surface the real error (e.g. quota exceeded, invalid key)
            friendly = str(exc)
            if "insufficient_quota" in friendly or "429" in friendly:
                friendly = "OpenAI quota exceeded — add billing credits at platform.openai.com or use a different API key."
            elif "invalid_api_key" in friendly or "401" in friendly:
                friendly = "Invalid OpenAI API key — update OPENAI_API_KEY in your .env file and restart."
            await self.websocket.send_json({"type": "error", "message": friendly})
        finally:
            self.session.transition(SessionState.IDLE)
            await self.websocket.send_json({"type": "response_complete"})

    async def _handle_direct_audio(self, message: dict) -> None:
        """
        Non-WebRTC fallback: browser sends PCM as binary WebSocket message.
        Used when WebRTC negotiation is unavailable.
        """
        router = await self._get_or_init_router()
        await router.process_audio_chunk(message["data"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _get_or_init_router(self):
        """Lazy-init an AudioRouter for the WebSocket-only fallback path."""
        if self.webrtc.router is None:
            from audio_router import AudioRouter

            self.webrtc.router = AudioRouter(self.session)
            await self.webrtc.router.load_vad()
        return self.webrtc.router

    async def _messages(self):
        """Async generator that yields parsed messages from the WebSocket."""
        while True:
            try:
                data = await self.websocket.receive()
                if "text" in data:
                    yield json.loads(data["text"])
                elif "bytes" in data:
                    # Raw binary audio (direct pathway)
                    yield {"type": "audio_data", "data": data["bytes"]}
            except Exception as exc:
                logger.info(f"[{self.session.session_id}] WebSocket closed: {exc}")
                break
