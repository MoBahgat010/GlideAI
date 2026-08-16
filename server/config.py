import os
from pathlib import Path
from dotenv import load_dotenv

_SERVER_DIR = Path(__file__).resolve().parent
_SERVER_ENV = _SERVER_DIR / ".env"
_ROOT_ENV = _SERVER_DIR.parent / ".env"

if _SERVER_ENV.exists():
    load_dotenv(_SERVER_ENV, override=True)
elif _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=True)
else:
    load_dotenv(override=True)

DEVICE = os.getenv("DEVICE", "cuda")

MAX_CHARS = int(os.getenv("MAX_CHARS"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL_D_MODEL = int(os.getenv("EMBEDDING_MODEL_D_MODEL"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

EMBED_BATCH = int(os.getenv("EMBED_BATCH"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K"))

INDEX_NAME = os.getenv("INDEX_NAME")

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
SUMMARIZER = os.getenv("SUMMARIZER")
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD"))
RUN_LIMIT = int(os.getenv("RUN_LIMIT"))
LVLM = os.getenv("LVLM")

HF_TOKEN = os.getenv("HF_TOKEN")

REDIS_URL = os.getenv("REDIS_URL")

REV_AI = os.getenv("REV_AI")

SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")

WEAVIATE_REST_ENDPOINT = os.getenv("WEAVIATE_REST_ENDPOINT")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER")

ENABLE_SEMANTIC_MEMORY = os.getenv("ENABLE_SEMANTIC_MEMORY", None)
ENABLE_EPISODIC_MEMORY = os.getenv("ENABLE_EPISODIC_MEMORY", None)

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "test_uploads/chunks")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "test_uploads/final")

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "graphrag_db")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS"))

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL")
TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL")
