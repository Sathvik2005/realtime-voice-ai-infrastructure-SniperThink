"""
Voice AI System — FastAPI Backend Entry Point

Bootstraps all ML models at startup, serves the frontend,
and exposes the WebSocket signaling endpoint per voice session.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

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
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Suppress overly verbose logs from third-party libraries in production
if log_level != "DEBUG":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
session_manager = SessionManager()
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Rate limiting: track requests per IP
rate_limit_storage: Dict[str, list] = {}
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 100))  # per minute
RATE_LIMIT_WINDOW = 60  # seconds


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
    docs_url="/docs" if os.environ.get("ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.environ.get("ENABLE_DOCS", "true").lower() == "true" else None,
)

# ---------------------------------------------------------------------------
# Middleware — Production Security & CORS
# ---------------------------------------------------------------------------

# CORS: Allow frontend from different origins in production
allowed_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host: Prevent host header attacks
trusted_hosts = os.environ.get("TRUSTED_HOSTS", "*").split(",")
if "*" not in trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiting per IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Clean old entries
    if client_ip in rate_limit_storage:
        rate_limit_storage[client_ip] = [
            ts for ts in rate_limit_storage[client_ip]
            if now - ts < RATE_LIMIT_WINDOW
        ]
    
    # Check rate limit
    if client_ip in rate_limit_storage:
        if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please try again later."},
            )
    
    # Record this request
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    rate_limit_storage[client_ip].append(now)
    
    response = await call_next(request)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for production."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc) if os.environ.get("DEBUG", "false").lower() == "true" else "Contact support"},
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
    return {
        "status": "ok",
        "active_sessions": session_manager.active_count(),
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
    }


@app.get("/readiness")
async def readiness_check():
    """Readiness probe — checks if models are loaded."""
    from audio_router import _llm, _tts
    
    ready = True
    components = {
        "llm": _llm is not None,
        "tts": _tts is not None,
    }
    
    if not all(components.values()):
        ready = False
    
    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if ready else "not_ready",
            "components": components,
        },
    )


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon to prevent 404 errors."""
    favicon_path = FRONTEND_DIR / "favicon.ico"
    if favicon_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(favicon_path)
    # Return a 204 No Content instead of 404 if favicon doesn't exist
    from fastapi.responses import Response
    return Response(status_code=204)


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
    workers = int(os.environ.get("WORKERS", 1))
    
    # Production uvicorn settings
    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": port,
        "log_level": os.environ.get("LOG_LEVEL", "info").lower(),
        "access_log": os.environ.get("ACCESS_LOG", "true").lower() == "true",
        "use_colors": True,
    }
    
    # Enable reload only in development
    if os.environ.get("ENVIRONMENT", "development") == "development":
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = [str(Path(__file__).parent)]
    
    logger.info(f"Starting Voice AI System on port {port}")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    uvicorn.run(**uvicorn_config)
