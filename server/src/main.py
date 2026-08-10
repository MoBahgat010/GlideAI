from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
from urllib import request

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import TRITON_HTTP_URL
from db.mongo import MongoManager
from db.redis import close_redis
from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.ingest import router as ingest_router
from routers.sessions import router as sessions_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server.main")


def check_triton_health(url: str) -> bool:
    """Check if Triton Inference Server is running and ready via HTTP GET."""
    health_url = f"{url}/v2/health/ready"
    try:
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle events."""
    logger.info("Initializing GraphRAG FastAPI Application...")

    # Check Triton Server status on startup; fail fast if Triton is down
    if not check_triton_health(TRITON_HTTP_URL):
        error_msg = f"❌ Triton Inference Server at '{TRITON_HTTP_URL}' is NOT running or ready! Please start Triton server first."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info("✅ Triton Inference Server health check passed (%s).", TRITON_HTTP_URL)

    try:
        await MongoManager.connect()
    except Exception as e:
        logger.warning("MongoDB connection setup during startup: %s", e)

    yield

    logger.info("Shutting down GraphRAG FastAPI Application...")
    await MongoManager.close()
    await close_redis()


app = FastAPI(
    title="GraphRAG Multimodal Agentic API",
    version="2.0.0",
    description="FastAPI Backend with MongoDB JWT Auth, Isolated Sessions, Agentic RAG, Redis Working Memory & Celery Memory Extraction.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/api/health")
async def health():
    """Readiness probe endpoint."""
    return {"status": "ok", "app": "GraphRAG Multimodal Agentic API", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "server.src.main:app",
        host="localhost",
        port=8000,
        reload=True,
        reload_excludes=["*.log", "test_uploads/*", "*.part", "__pycache__/*"],
    )
