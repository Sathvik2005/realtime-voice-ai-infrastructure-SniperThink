"""
WebRTC Server (aiortc)

Manages one RTCPeerConnection per session.
Receives the browser's microphone audio track and routes each
decoded frame through the processing pipeline.
"""

import asyncio
import logging
from typing import Optional

import numpy as np

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.exceptions import MediaStreamError
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("aiortc not installed — WebRTC transport disabled. "
                       "Falling back to WebSocket audio path.")

logger = logging.getLogger(__name__)

# WebRTC delivers audio at 48 kHz; VAD/ASR expect 16 kHz
WEBRTC_SAMPLE_RATE = 48_000
TARGET_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Frame converter (runs in hot path — keep allocation minimal)
# ---------------------------------------------------------------------------
def convert_audio_frame(frame) -> Optional[bytes]:
    """
    Convert an aiortc AudioFrame (48 kHz, possibly stereo, s16/fltp) to
    16 kHz mono int16 PCM bytes suitable for Silero VAD and Whisper.
    """
    try:
        audio = frame.to_ndarray()  # shape: (channels, samples) or (samples,)

        # Flatten to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Normalise to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32_768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2_147_483_648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Linear-interpolation downsample to 16 kHz
        src_rate = frame.sample_rate
        if src_rate != TARGET_SAMPLE_RATE:
            new_len = int(len(audio) * TARGET_SAMPLE_RATE / src_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )

        # Back to int16 bytes
        return (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16).tobytes()

    except Exception as exc:
        logger.error(f"Frame conversion error: {exc}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Continuous track consumer
# ---------------------------------------------------------------------------
async def _consume_audio_track(track, router) -> None:
    """
    Drain an audio MediaStreamTrack in a tight loop, converting each frame
    and forwarding it to the AudioRouter.
    """
    while True:
        try:
            frame = await track.recv()
            pcm = convert_audio_frame(frame)
            if pcm:
                await router.process_audio_chunk(pcm)
        except MediaStreamError:
            logger.info("Audio track ended (MediaStreamError)")
            break
        except Exception as exc:
            logger.warning(f"Track consumer error: {exc}")
            break


# ---------------------------------------------------------------------------
# WebRTC server
# ---------------------------------------------------------------------------
class WebRTCServer:
    """
    One instance per session.  Creates the RTCPeerConnection,
    processes the SDP offer/answer, and wires the inbound audio
    track to the AudioRouter.
    """

    def __init__(self, session) -> None:
        self.session = session
        self.pc: Optional[object] = None  # RTCPeerConnection
        self.router = None                # AudioRouter (set on first offer)

    # ------------------------------------------------------------------
    # SDP negotiation
    # ------------------------------------------------------------------
    async def handle_offer(self, sdp: str, type_: str) -> dict:
        """
        Accept a browser SDP offer and return an SDP answer.
        Also instantiates the AudioRouter for this session.
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc is not installed; WebRTC is unavailable.")

        # Lazy-init AudioRouter
        from audio_router import AudioRouter

        self.router = AudioRouter(self.session)
        await self.router.load_vad()

        self.pc = RTCPeerConnection()
        self.session.peer_connection = self.pc

        @self.pc.on("track")
        async def on_track(track):
            logger.info(
                f"[{self.session.session_id}] Received {track.kind} track"
            )
            if track.kind == "audio":
                asyncio.create_task(
                    _consume_audio_track(track, self.router),
                    name=f"audio-{self.session.session_id}",
                )

        @self.pc.on("connectionstatechange")
        async def on_state_change():
            logger.info(
                f"[{self.session.session_id}] WebRTC state: {self.pc.connectionState}"
            )
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=type_)
        )
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
        }

    # ------------------------------------------------------------------
    # ICE
    # ------------------------------------------------------------------
    async def add_ice_candidate(self, candidate: dict) -> None:
        if self.pc is None:
            return
        ice = RTCIceCandidate(
            sdpMid=candidate.get("sdpMid"),
            sdpMLineIndex=candidate.get("sdpMLineIndex"),
            candidate=candidate.get("candidate", ""),
        )
        await self.pc.addIceCandidate(ice)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def close(self) -> None:
        if self.pc is not None:
            try:
                await self.pc.close()
            except Exception:
                pass
