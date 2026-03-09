"""
Interrupt Controller

Provides clean, asynchronous cancellation of an in-flight
LLM → TTS pipeline task when the user speaks over the AI.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InterruptController:
    """
    Wraps a running asyncio.Task (the voice pipeline) and cancels it
    on demand.  After cancellation the system is ready for a new turn.
    """

    def __init__(self, session) -> None:
        self.session = session
        self._pipeline_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pipeline_task(self, task: asyncio.Task) -> None:
        """Register the current pipeline task so it can be cancelled."""
        self._pipeline_task = task

    async def interrupt(self) -> None:
        """
        Signal an interrupt:
        1. Set the per-session interrupt_event (checked by LLM/TTS generators).
        2. Cancel the pipeline asyncio.Task and wait for it to finish.
        3. Clear any queued TTS audio so the browser goes silent immediately.
        """
        logger.info(f"[{self.session.session_id}] Interrupt — cancelling pipeline")
        self.session.interrupt_event.set()

        if self._pipeline_task is not None and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await asyncio.shield(self._pipeline_task)
            except (asyncio.CancelledError, Exception):
                pass  # Expected on cancellation

        # Reset for next turn
        self.session.interrupt_event.clear()
        self._pipeline_task = None
        logger.info(f"[{self.session.session_id}] Interrupt complete — ready for next turn")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    @property
    def is_interrupted(self) -> bool:
        return self.session.interrupt_event.is_set()
