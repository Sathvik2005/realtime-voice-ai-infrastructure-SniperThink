"""
Voice Activity Detection (VAD)  —  Silero VAD wrapper

Processes a continuous stream of 16 kHz int16 PCM audio and emits
two events:
  • "speech_start"  — user began speaking
  • "speech_end"    — user finished speaking (triggers ASR)

The underlying Silero model is loaded once as a class-level variable
and shared across all per-session VADEngine instances.
"""

import asyncio
import logging
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512          # 32 ms per Silero frame @ 16 kHz
SPEECH_THRESHOLD = 0.50      # probability above → speech
SILENCE_THRESHOLD = 0.35     # probability below → silence

# Speech must persist for N frames before "speech_start" fires
MIN_SPEECH_FRAMES = 3        # ≈ 96 ms
# Silence must persist for N frames before "speech_end" fires
MIN_SILENCE_FRAMES = 15      # ≈ 480 ms


class VADEngine:
    """
    Per-session VAD state machine backed by a shared Silero model.
    Thread-safe: never mutates class-level state.
    """

    _model = None  # Shared Silero model (loaded once at startup)

    # ------------------------------------------------------------------
    # Class-level model loading
    # ------------------------------------------------------------------
    @classmethod
    async def load_shared_model(cls) -> None:
        if cls._model is not None:
            return
        loop = asyncio.get_event_loop()
        cls._model = await loop.run_in_executor(None, cls._load_silero)
        logger.info("Silero VAD model loaded.")

    @staticmethod
    def _load_silero():
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Instance initialisation
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        # Per-session state
        self._speech_active: bool = False
        self._speech_frames: int = 0
        self._silence_frames: int = 0
        self._leftover: np.ndarray = np.array([], dtype=np.float32)

        # Async callbacks (set by AudioRouter)
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def process_chunk(self, audio_bytes: bytes) -> List[str]:
        """
        Feed a chunk of 16 kHz int16 PCM audio.
        Returns a list of events fired during this chunk (may be empty).
        """
        samples = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32_768.0
        )

        # Prepend leftover from previous call
        if len(self._leftover) > 0:
            samples = np.concatenate([self._leftover, samples])

        events: List[str] = []
        i = 0
        while i + FRAME_SAMPLES <= len(samples):
            frame = samples[i : i + FRAME_SAMPLES]
            prob = self._infer(frame)
            event = await self._update_state(prob)
            if event:
                events.append(event)
            i += FRAME_SAMPLES

        self._leftover = samples[i:].copy()
        return events

    def reset(self) -> None:
        self._speech_active = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._leftover = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _infer(self, frame: np.ndarray) -> float:
        if self._model is None:
            return 0.0
        import torch
        tensor = torch.from_numpy(frame).unsqueeze(0)
        with torch.no_grad():
            return self._model(tensor, SAMPLE_RATE).item()

    async def _update_state(self, prob: float) -> Optional[str]:
        if not self._speech_active:
            if prob >= SPEECH_THRESHOLD:
                self._speech_frames += 1
                self._silence_frames = 0
                if self._speech_frames >= MIN_SPEECH_FRAMES:
                    self._speech_active = True
                    self._speech_frames = 0
                    logger.debug("VAD → speech_start")
                    if self.on_speech_start:
                        await self.on_speech_start()
                    return "speech_start"
            else:
                self._speech_frames = 0
        else:
            if prob < SILENCE_THRESHOLD:
                self._silence_frames += 1
                self._speech_frames = 0
                if self._silence_frames >= MIN_SILENCE_FRAMES:
                    self._speech_active = True  # keep True so buffer continues until pipeline drains
                    self._silence_frames = 0
                    logger.debug("VAD → speech_end")
                    if self.on_speech_end:
                        await self.on_speech_end()
                    self._speech_active = False
                    return "speech_end"
            else:
                self._silence_frames = 0
                self._speech_frames += 1

        return None
