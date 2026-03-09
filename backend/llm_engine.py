"""
LLM Engine  —  streaming OpenAI / local model responses

Maintains no mutable state between calls; conversation history is
passed in per-request and ownership stays with the session.

Streaming design:
  stream_response() is an async generator that yields text delta
  strings as they arrive.  Callers can pipe directly into TTS.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Respond in 1–3 natural sentences suitable for text-to-speech. "
    "Avoid bullet points, markdown, or lists."
)
MAX_HISTORY_TURNS = 10   # Older turns are dropped to stay within context limits
MAX_TOKENS = 300


class LLMEngine:
    """
    Thin async wrapper around OpenAI chat completions with streaming.
    Swap the provider easily by extending _stream_tokens().
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    async def initialize(self) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=self._api_key)
        logger.info(f"LLM initialised: model={self.model}")

    # ------------------------------------------------------------------
    # Streaming response  (main public API)
    # ------------------------------------------------------------------
    async def stream_response(
        self,
        transcript: str,
        history: List[Dict],
        context: str = "",
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator yielding text chunks as they arrive from the LLM.
        Stops early if interrupt_event is set.
        """
        messages = self._build_messages(transcript, history, context)
        async for token in self._stream_tokens(messages, interrupt_event):
            yield token

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    @staticmethod
    def update_history(
        history: List[Dict], user_text: str, assistant_text: str
    ) -> List[Dict]:
        history = list(history)  # never mutate caller's list
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        # Trim to MAX_HISTORY_TURNS turns (each turn = 2 messages)
        max_messages = MAX_HISTORY_TURNS * 2
        if len(history) > max_messages:
            history = history[-max_messages:]
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_messages(
        self, transcript: str, history: List[Dict], context: str
    ) -> List[Dict]:
        messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append(
                {"role": "system", "content": f"Relevant context:\n{context}"}
            )

        # Include recent history (already trimmed)
        messages.extend(history[-MAX_HISTORY_TURNS * 2 :])
        messages.append({"role": "user", "content": transcript})
        return messages

    async def _stream_tokens(
        self,
        messages: List[Dict],
        interrupt_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[str, None]:
        if self._client is None:
            yield "I'm not configured yet. Please set OPENAI_API_KEY."
            return

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )
            async for chunk in stream:
                if interrupt_event and interrupt_event.is_set():
                    await stream.close()
                    logger.info("LLM stream cancelled by interrupt")
                    return
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error(f"LLM streaming error: {exc}", exc_info=True)
            raise  # let the caller handle and surface the real message
