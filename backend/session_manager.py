"""
Session Manager

Tracks all active voice sessions.  Each session is an independent
state machine that walks through the voice pipeline stages.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class SessionState(Enum):
    IDLE = auto()        # No activity
    LISTENING = auto()   # User is speaking — buffering audio
    PROCESSING = auto()  # Transcribing / awaiting LLM response
    SPEAKING = auto()    # TTS audio streaming to client


# ---------------------------------------------------------------------------
# Session data class
# ---------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str

    # Conversation state
    state: SessionState = SessionState.IDLE
    conversation_history: list = field(default_factory=list)

    # Async coordination
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Runtime references (set after connection)
    peer_connection: object = field(default=None, repr=False)
    websocket: object = field(default=None, repr=False)

    def transition(self, new_state: SessionState) -> None:
        """Log and apply a state transition."""
        if self.state != new_state:
            logger.info(f"[{self.session_id}] {self.state.name} → {new_state.name}")
            self.state = new_state


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def shutdown(self) -> None:
        """Close all open peer connections."""
        async with self._lock:
            for session in list(self._sessions.values()):
                if session.peer_connection is not None:
                    try:
                        await session.peer_connection.close()
                    except Exception:
                        pass
            self._sessions.clear()
        logger.info("SessionManager: all sessions closed.")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    async def create_session(self, session_id: str) -> Session:
        async with self._lock:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            logger.info(f"Session created: {session_id}")
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None and session.peer_connection is not None:
                try:
                    await session.peer_connection.close()
                except Exception:
                    pass
            logger.info(f"Session removed: {session_id}")

    def active_count(self) -> int:
        return len(self._sessions)
