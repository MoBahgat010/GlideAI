"""
RAG pipeline configuration — Pinecone, NVIDIA NIM, Redis, object storage.
Independent of the old GraphRAG / Neo4j config.py.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise EnvironmentError(f"Missing required env var: {name}")
    return v


# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY = _require("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("INDEX_NAME", "rag-pipeline")

# ── NVIDIA NIM ────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = _require("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
LOCAL_LLM_URL = os.getenv("QWEN_SERVER_URL", "http://127.0.0.1:8080/v1")
LIGHT_WEIGHT_MODEL = os.getenv("LIGHT_WEIGHT", "openai/gpt-oss-120b")
HEAVY_WEIGHT_MODEL = os.getenv("HEAVY_WEIGHT", "thinkingmachines/inkling")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-clip-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# ── OpenDataLoader ────────────────────────────────────────────────────────────
HYBRID_URL = os.getenv("HYBRID_URL", "http://localhost:5002")

# ── Rev AI Speech-to-Text ─────────────────────────────────────────────────────
REV_AI = os.getenv("REV_AI")
REV_AI_POLL_SECONDS = int(os.getenv("REV_AI_POLL_SECONDS", "10"))
STT_SEGMENT_SECONDS = int(os.getenv("STT_SEGMENT_SECONDS", "60"))

# ── Celery / Redis ────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Object storage ────────────────────────────────────────────────────────────
S3_URI = os.getenv("S3_URI")                                   # None → disk
UPLOAD_DIR_RAW = os.getenv("UPLOAD_DIR", "test")
UPLOAD_DIR = str(BASE_DIR / UPLOAD_DIR_RAW) if not Path(UPLOAD_DIR_RAW).is_absolute() else UPLOAD_DIR_RAW

IMAGE_DIR_RAW = os.getenv("IMAGE_DIR", "ingested_images")
IMAGE_DIR = str(BASE_DIR / IMAGE_DIR_RAW) if not Path(IMAGE_DIR_RAW).is_absolute() else IMAGE_DIR_RAW

# ── Pipeline tuning ───────────────────────────────────────────────────────────
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "30"))
RERANK_TOP_K   = int(os.getenv("RERANK_TOP_K", "5"))
EMBED_BATCH    = int(os.getenv("EMBED_BATCH", "64"))
PDF_PAGE_CHUNK_SIZE = int(os.getenv("PDF_PAGE_CHUNK_SIZE", "2"))

# ─ Weaviate ────────────────────────────────────────────────────────────────
WEAVIATE_REST_ENDPOINT = os.getenv("WEAVIATE_REST_ENDPOINT")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
