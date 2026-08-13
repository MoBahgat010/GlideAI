import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEVICE = os.getenv("DEVICE", "cuda")

MAX_CHARS = int(os.getenv("MAX_CHARS", "1000"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "200"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/siglip-large-patch16-384")
EMBEDDING_MODEL_D_MODEL = int(os.getenv("EMBEDDING_MODEL_D_MODEL", "768"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-m0")

INDEX_NAME = os.getenv("INDEX_NAME", "personal-trials")

BASE_URL = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("API_KEY", "")
SUMMARIZER = os.getenv("SUMMARIZER", "openai/gpt-oss-120b")
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD", "100000"))
RUN_LIMIT = int(os.getenv("RUN_LIMIT", "30"))
LVLM = os.getenv("LVLM", "nvidia/nemotron-3-ultra-550b-a55b")

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
TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "localhost:8000")
TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")