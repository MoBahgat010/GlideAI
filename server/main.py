import sys
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from urllib import request
import os
_SERVER_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SERVER_DIR.parent
_SRC_DIR = _SERVER_DIR / "src"

for p in [str(_ROOT_DIR), str(_SRC_DIR), str(_SERVER_DIR)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from RAG_Pipeline.ingestion.execute import vdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import TRITON_HTTP_URL, UPLOAD_DIR, CHUNKS_DIR

from src.db.mongo import MongoManager
from src.db.redis import close_redis
from src.routers.auth import router as auth_router
from src.routers.chat import router as chat_router, agent_runner
from src.routers.ingest import router as ingest_router
from src.routers.sessions import router as sessions_router
from src.routers.google_auth import router as google_auth_router
from src.routers.gmail import router as gmail_router
from src.interrupts.router import router as hitl_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server.main")


def check_triton_health(url: str) -> bool:
    if not url.startswith("http"):
        url = f"http://{url}"
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
    logger.info("Initializing Enterprise RAG Multi-Modal Platform...")

    triton_ready = check_triton_health(TRITON_HTTP_URL)
    if triton_ready:
        logger.info("Triton Inference Server health check passed (%s).", TRITON_HTTP_URL)
    else:
        logger.warning("Triton Inference Server at '%s' is not ready yet. Inference will connect dynamically.", TRITON_HTTP_URL)

    try:
        await MongoManager.connect()
    except Exception as e:
        logger.warning("MongoDB connection setup during startup: %s", e)

    yield

    logger.info("Shutting down Enterprise RAG Platform...")
    await MongoManager.close()
    await close_redis()
    vdb.close()


app = FastAPI(
    title="Enterprise Multi-Modal Agentic GraphRAG API",
    version="2.0.0",
    description="Production Multi-Modal RAG Platform with Triton GPU Acceleration, Hybrid Search, JWT Refresh Token Rotation, Redis Working Memory, and Cloudinary CDN Asset Integration.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(google_auth_router)
app.include_router(gmail_router)
app.include_router(hitl_router)
app.include_router(sessions_router)
app.include_router(ingest_router)
app.include_router(chat_router)

@app.get("/api/health")
async def health():
    """Readiness probe endpoint."""
    return {
        "status": "ok",
        "platform": "Enterprise Multi-Modal Agentic GraphRAG",
        "version": "2.0.0",
        "triton_ready": check_triton_health(TRITON_HTTP_URL),
    }


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_excludes=["*.log", "test_uploads/*", "*.part", "__pycache__/*"],
    )
