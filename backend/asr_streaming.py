"""
Streaming Speech Recognition  —  Faster-Whisper

Each voice session owns one StreamingASR instance that maintains
an audio buffer for the current utterance.  The underlying
faster_whisper.WhisperModel is shared (class variable) to avoid
loading it multiple times.

Transcription strategy:
  • partial()  — fast, low‐beam decode for live display (called ~every 500 ms)
  • final()    — accurate, higher-beam decode on full utterance, clears buffer
"""

import asyncio
import logging
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
