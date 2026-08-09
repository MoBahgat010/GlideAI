import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEVICE = os.getenv("DEVICE", "cuda")

MAX_CHARS = int(os.getenv("MAX_CHARS", "1000"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "200"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-clip-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-m0")

INDEX_NAME = os.getenv("INDEX_NAME", "personal-trials")
QWEN_SERVER_URL = os.getenv("QWEN_SERVER_URL", "http://127.0.0.1:8080")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen")


BASE_URL = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
LIGHT_WEIGHT = os.getenv("LIGHT_WEIGHT", "openai/gpt-oss-120b")
HEAVY_WEIGHT = os.getenv("HEAVY_WEIGHT", "nvidia/nemotron-3-ultra-550b-a55b")

HF_TOKEN = os.getenv("HF_TOKEN", "")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6000/0")

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "50"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))

REV_AI = os.getenv("REV_AI", "")
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY", "")

WEAVIATE_REST_ENDPOINT = os.getenv("WEAVIATE_REST_ENDPOINT", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")


# Temp ones
CHUNKS_DIR = "test_uploads/chunks"
UPLOAD_DIR = "test_uploads/final"

# ── MongoDB & JWT ──────────────────────────────────────────────────────────────
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "graphrag_db")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "graphrag_super_secret_jwt_key_2026_change_in_production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# ── Triton Inference Server ────────────────────────────────────────────────────
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")