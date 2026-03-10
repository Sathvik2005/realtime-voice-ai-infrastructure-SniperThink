"""
LLM Engine  —  streaming responses with automatic fallback chain

Fallback chain (tried in order, first that succeeds wins):
  1. OpenAI API       — primary; best quality, cloud-hosted
  2. Ollama local LLM — secondary; fully offline, runs on localhost:11434
  3. Rule-based       — tertiary; always works, no dependencies whatsoever

This means the system remains fully usable even when:
  • OPENAI_API_KEY is missing or quota is exhausted
  • Ollama is not installed
  • There is no internet connectivity

Streaming design:
  stream_response() is an async generator that yields text delta
  strings as they arrive.  Callers can pipe directly into TTS.
  Each provider implements _stream_tokens() so they are swappable.
"""

import asyncio
import logging
import os
import re
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Respond in 1–3 natural sentences suitable for text-to-speech. "
    "Avoid bullet points, markdown, or lists."
)
MAX_HISTORY_TURNS = 10
MAX_TOKENS = 300

# ---------------------------------------------------------------------------
# Rule-based fallback — no external dependencies
# ---------------------------------------------------------------------------

_RULES: List[tuple] = [
    (r"\bhello\b|\bhi\b|\bhey\b",
     "Hello! I'm your voice assistant. How can I help you today?"),
    (r"\bwhat.*(your name|are you|who are you)\b",
     "I'm a voice AI assistant built on a real-time streaming pipeline."),
    (r"\bhow are you\b",
     "I'm running well, thank you for asking! What can I do for you?"),
    (r"\bthank(s| you)\b",
     "You're welcome! Is there anything else I can help with?"),
    (r"\bbye\b|\bgoodbye\b|\bsee you\b",
     "Goodbye! Have a great day."),
    (r"\bweather\b",
     "I don't have access to live weather data right now, but you can check a weather website for the latest forecast."),
    (r"\btime\b|\bclock\b",
     "I don't have access to a real-time clock in this mode, but your device should show the current time."),
    (r"\bhelp\b|\bwhat can you do\b",
     "I can answer questions and have a conversation with you. My AI backend may be in fallback mode right now, so complex questions might get simple answers."),
    (r"\bsorry\b|\bapolog\b",
     "No worries at all! What would you like to talk about?"),
]

def _rule_based_response(text: str) -> str:
    """Return a scripted reply if any keyword rule matches, else a generic reply."""
    lower = text.lower()
    for pattern, reply in _RULES:
        if re.search(pattern, lower):
            return reply
    return (
        "I heard you, but my AI backend is temporarily unavailable. "
        "Please try again in a moment, or check that your API key or Ollama service is configured."
    )


# ---------------------------------------------------------------------------
# Provider: OpenAI
# ---------------------------------------------------------------------------

class _OpenAIProvider:
    name = "OpenAI"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key
        self._client = None

    async def initialize(self) -> bool:
        """Return True if the client can be instantiated."""
        if not self._api_key:
            logger.warning("OpenAI: no API key configured — skipping")
            return False
        try:
            import openai
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
            logger.info(f"OpenAI provider ready: model={self.model}")
            return True
        except ImportError:
            logger.warning("OpenAI: openai package not installed — skipping")
            return False

    async def stream_tokens(
        self,
        messages: List[Dict],
        interrupt_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[str, None]:
        if self._client is None:
            raise RuntimeError("OpenAI client not initialised")
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
                logger.info("OpenAI stream cancelled by interrupt")
                return
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Provider: Ollama  (local LLM via REST)
# ---------------------------------------------------------------------------

class _OllamaProvider:
    name = "Ollama"

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available = False

    async def initialize(self) -> bool:
        """Return True if Ollama is reachable and the model exists."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    logger.warning(f"Ollama: /api/tags returned {resp.status_code}")
                    return False
                tags = resp.json().get("models", [])
                names = [m.get("name", "").split(":")[0] for m in tags]
                model_base = self.model.split(":")[0]
                if model_base not in names:
                    logger.warning(
                        f"Ollama: model '{self.model}' not found. "
                        f"Available: {names}. "
                        f"Run: ollama pull {self.model}"
                    )
                    # Still mark available — user can pull later; we'll error at call time
                self._available = True
                logger.info(f"Ollama provider ready: model={self.model} at {self.base_url}")
                return True
        except ImportError:
            logger.warning("Ollama: httpx not installed — skipping")
            return False
        except Exception as exc:
            logger.warning(f"Ollama: not reachable ({exc}) — skipping")
            return False

    async def stream_tokens(
        self,
        messages: List[Dict],
        interrupt_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[str, None]:
        import httpx
        import json as _json

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": MAX_TOKENS},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if interrupt_event and interrupt_event.is_set():
                        logger.info("Ollama stream cancelled by interrupt")
                        return
                    if not line.strip():
                        continue
                    try:
                        data = _json.loads(line)
                    except Exception:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        return


# ---------------------------------------------------------------------------
# Provider: Rule-based  (always available)
# ---------------------------------------------------------------------------

class _RuleBasedProvider:
    name = "RuleBased"

    async def initialize(self) -> bool:
        logger.info("Rule-based fallback provider ready (always active)")
        return True

    async def stream_tokens(
        self,
        messages: List[Dict],
        interrupt_event: Optional[asyncio.Event],
    ) -> AsyncGenerator[str, None]:
        # Extract the last user message
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        reply = _rule_based_response(user_text)
        # Simulate token streaming with word-level chunks
        for word in reply.split():
            if interrupt_event and interrupt_event.is_set():
                return
            yield word + " "
            await asyncio.sleep(0.02)   # ~50 tokens/sec feel


# ---------------------------------------------------------------------------
# LLMEngine  —  public interface with automatic fallback
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Orchestrates the provider fallback chain.

    At initialization the engine probes each provider and builds an ordered
    list of those that are available.  stream_response() tries them in order;
    if a provider raises, it logs the error and moves to the next one.
    The rule-based provider is always last and never fails.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        ollama_model: str = "llama3",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._providers: list = [
            _OpenAIProvider(model, api_key or os.environ.get("OPENAI_API_KEY", "")),
            _OllamaProvider(ollama_model, ollama_url),
            _RuleBasedProvider(),
        ]
        self._active_providers: list = []
        self._active_provider_name: str = "none"

    async def initialize(self) -> None:
        """Probe all providers and record which ones are usable."""
        for p in self._providers:
            try:
                ok = await p.initialize()
                if ok:
                    self._active_providers.append(p)
            except Exception as exc:
                logger.warning(f"Provider {p.name} init error: {exc}")

        names = [p.name for p in self._active_providers]
        self._active_provider_name = names[0] if names else "none"
        logger.info(f"LLM initialised: model={self.model}  chain={names}")

    @property
    def active_provider(self) -> str:
        return self._active_provider_name

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
        Async generator yielding text chunks.
        Tries providers in order; falls back silently on failure.
        """
        messages = self._build_messages(transcript, history, context)

        for provider in self._active_providers:
            try:
                logger.debug(f"LLM: trying provider {provider.name}")
                got_any = False
                async for token in provider.stream_tokens(messages, interrupt_event):
                    got_any = True
                    self._active_provider_name = provider.name
                    yield token
                if got_any:
                    return   # success — stop here
                # Provider returned empty stream — try next
                logger.warning(f"LLM: {provider.name} returned empty stream, trying next")
            except Exception as exc:
                logger.warning(f"LLM: {provider.name} failed ({exc}), trying next provider")
                continue

        # All providers exhausted (should never happen — RuleBased always works)
        yield "I'm sorry, I couldn't process your request right now. Please try again."

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    @staticmethod
    def update_history(
        history: List[Dict], user_text: str, assistant_text: str
    ) -> List[Dict]:
        history = list(history)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
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

        messages.extend(history[-MAX_HISTORY_TURNS * 2:])
        messages.append({"role": "user", "content": transcript})
        return messages
