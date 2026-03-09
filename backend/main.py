"""
Voice AI System — FastAPI Backend Entry Point

Bootstraps all ML models at startup, serves the frontend,
and exposes the WebSocket signaling endpoint per voice session.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Load .env from project root before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv optional; env vars may be set by the OS

# Allow sibling packages (rag/) to be imported from the backend process
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_router import initialize_models
from session_manager import SessionManager
from websocket_signaling import SignalingHandler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
session_manager = SessionManager()
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models before accepting connections; clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("  Voice AI System  —  starting up")
    logger.info("=" * 60)

    try:
        await initialize_models()
        logger.info("All models ready — accepting connections.")
    except Exception as exc:
        logger.error(f"Model initialisation failed: {exc}")
        logger.warning("Starting in degraded mode (some features may be unavailable).")

    yield  # Application runs here

    logger.info("Shutting down Voice AI System …")
    await session_manager.shutdown()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice AI System",
    description="Real-time bidirectional voice conversation pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Voice AI System</h1><p>Frontend not found.</p>", status_code=404)


@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "active_sessions": session_manager.active_count()}


# ---------------------------------------------------------------------------
# WebSocket endpoint  — one connection per voice session
# ---------------------------------------------------------------------------
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Handles signaling (SDP offer/answer, ICE candidates) and control
    messages (interrupt, status events) for a single voice session.

    Audio travels over WebRTC; this channel is signalling-only.
    TTS audio chunks are streamed back here as binary frames.
    """
    session = await session_manager.create_session(session_id)
    handler = SignalingHandler(session=session, websocket=websocket)

    try:
        await websocket.accept()
        logger.info(f"[{session_id}] Client connected")
        await handler.run()
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as exc:
        logger.error(f"[{session_id}] Unexpected session error: {exc}", exc_info=True)
    finally:
        await session_manager.remove_session(session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
