import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Device
DEVICE = os.getenv("DEVICE", "cuda")

# Chunking
MAX_CHARS = int(os.getenv("MAX_CHARS"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS"))

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL_D_MODEL = int(os.getenv("EMBEDDING_MODEL_D_MODEL"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

# Reranker / Embed batch
EMBED_BATCH = int(os.getenv("EMBED_BATCH"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K"))

# Index Name
INDEX_NAME = os.getenv("INDEX_NAME")

# AI Provider & Summarizer
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
SUMMARIZER = os.getenv("SUMMARIZER")
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD"))
RUN_LIMIT = int(os.getenv("RUN_LIMIT"))
LVLM = os.getenv("LVLM")

# HF Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Redis
REDIS_URL = os.getenv("REDIS_URL")

# AI Services
REV_AI = os.getenv("REV_AI")

# MCP Provider
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")

# Weaviate
WEAVIATE_REST_ENDPOINT = os.getenv("WEAVIATE_REST_ENDPOINT")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Cloudinary
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER")

# Feature Flags
ENABLE_SEMANTIC_MEMORY = os.getenv("ENABLE_SEMANTIC_MEMORY", None)
ENABLE_EPISODIC_MEMORY = os.getenv("ENABLE_EPISODIC_MEMORY", None)
ENABLE_WORKING_MEMORY = os.getenv("ENABLE_WORKING_MEMORY", None)

# Storage Dirs
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "test_uploads/chunks")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "test_uploads/final")

# MongoDB
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "graphrag_db")

# JWT
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS"))

# Triton
TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL")
TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL")
