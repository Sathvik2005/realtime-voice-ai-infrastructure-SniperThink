"""
Streaming Text-to-Speech  —  Piper TTS (+ gTTS fallback)

Key design: sentence-boundary streaming.
As LLM tokens arrive, we buffer them.  The moment we detect a
sentence boundary (. ! ? newline) we ship that sentence to Piper
immediately, so audio playback starts well before the LLM is done.

Audio output format: raw 16-bit LE PCM at Piper's native rate (22 050 Hz).
The browser AudioContext resamples on playback.
"""

import asyncio
import logging
import os
import shutil
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

AUDIO_CHUNK_BYTES = 4_096   # ~93 ms of audio per yield at 22 050 Hz mono s16
SENTENCE_ENDS = frozenset(".!?\n")


class StreamingTTS:
    """
    Converts a streaming text generator into a streaming audio byte generator.

    Usage:
        async for audio_chunk in tts.synth_stream(text_gen, interrupt_event):
            await send(audio_chunk)
    """

    def __init__(self, voice: str = "en_US-lessac-medium") -> None:
        self.voice = voice
        self._piper_exe = self._find_piper()
        self._model_path = self._find_model(voice)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def synth_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Consume streaming text, synthesise audio sentence-by-sentence,
        and yield raw PCM bytes as they are produced.
        """
        sentence_buf = ""

        async for token in text_stream:
            if self._interrupted(interrupt_event):
                return

            sentence_buf += token

            # Check for sentence boundary
            boundary = -1
            for i, ch in enumerate(sentence_buf):
                if ch in SENTENCE_ENDS:
                    boundary = i
                    break

            if boundary >= 0:
                sentence = sentence_buf[: boundary + 1].strip()
                sentence_buf = sentence_buf[boundary + 1 :]
                if sentence:
                    async for chunk in self._synth_sentence(sentence, interrupt_event):
                        yield chunk

        # Flush any remaining text
        if sentence_buf.strip() and not self._interrupted(interrupt_event):
            async for chunk in self._synth_sentence(sentence_buf.strip(), interrupt_event):
                yield chunk

    async def synth_full(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Convenience wrapper for synthesising a complete string."""
        async def _gen():
            yield text

        async for chunk in self.synth_stream(_gen(), interrupt_event):
            yield chunk

    # ------------------------------------------------------------------
    # Synthesis backends
    # ------------------------------------------------------------------
    async def _synth_sentence(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[bytes, None]:
        logger.debug(f"TTS synthesising: '{text[:60]}'")
        if self._piper_exe and self._model_path and os.path.exists(self._model_path):
            async for chunk in self._piper_synth(text, interrupt_event):
                yield chunk
        else:
            async for chunk in self._gtts_synth(text):
                yield chunk

    async def _piper_synth(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[bytes, None]:
        """Stream raw PCM from Piper subprocess."""
        config_path = self._model_path + ".json"
        cmd = [
            self._piper_exe,
            "--model", self._model_path,
            "--output-raw",
        ]
        if os.path.exists(config_path):
            cmd += ["--config", config_path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            proc.stdin.write(text.encode())
            proc.stdin.close()

            while True:
                if self._interrupted(interrupt_event):
                    proc.kill()
                    return
                chunk = await proc.stdout.read(AUDIO_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk

            await proc.wait()

        except FileNotFoundError:
            logger.warning("Piper executable not found — falling back to gTTS")
            async for chunk in self._gtts_synth(text):
                yield chunk
        except Exception as exc:
            logger.error(f"Piper error: {exc}")

    async def _gtts_synth(self, text: str) -> AsyncGenerator[bytes, None]:
        """Fallback: synthesise via gTTS (requires internet, returns MP3)."""
        try:
            import io
            from gtts import gTTS

            loop = asyncio.get_event_loop()

            def _generate():
                tts = gTTS(text=text, lang="en", slow=False)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                return buf.getvalue()

            data = await loop.run_in_executor(None, _generate)
            for i in range(0, len(data), AUDIO_CHUNK_BYTES):
                yield data[i : i + AUDIO_CHUNK_BYTES]
        except Exception as exc:
            logger.error(f"gTTS fallback error: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _interrupted(event: Optional[asyncio.Event]) -> bool:
        return event is not None and event.is_set()

    @staticmethod
    def _find_piper() -> Optional[str]:
        for candidate in ["piper", "./piper/piper", "/usr/local/bin/piper"]:
            if shutil.which(candidate) or os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _find_model(voice: str) -> Optional[str]:
        candidates = [
            f"models/{voice}.onnx",
            f"./piper/voices/{voice}.onnx",
            f"/usr/share/piper-voices/{voice}.onnx",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return f"models/{voice}.onnx"  # Expected path; not checked at init
