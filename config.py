"""
Shared configuration and client setup for the GraphRAG pipeline.
Imported by both ingest.py and app.py so settings stay in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts" / "graphrag"
BATCH_DIR = ARTIFACT_DIR / "batch_jobs"

load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(override=True)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Set {name} in your .env before running this script.")
    return value


# --------------------------------------------------------------------------
# Neo4j
# --------------------------------------------------------------------------
NEO4J_URI = require_env("NEO4J_URI")
NEO4J_USERNAME = require_env("NEO4J_USERNAME")
NEO4J_PASSWORD = require_env("NEO4J_PASSWORD")

# --------------------------------------------------------------------------
# Gemini — used ONLY for the batch extraction job in ingest.py
# --------------------------------------------------------------------------
GEMINI_API_KEY = require_env("GEMINI_API_KEY")
GOOGLE_API_KEY = GEMINI_API_KEY
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemini-2.5-flash")
GEMINI_EXTRACTION_MODEL = EXTRACTION_MODEL

# --------------------------------------------------------------------------
# Local model — used for EVERYTHING else (summarizing communities, answering)
# --------------------------------------------------------------------------
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:14b-instruct")
LOCAL_BASE_URL = os.getenv("QWEN_SERVER_URL", "http://127.0.0.1:11434/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "local")

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

# --------------------------------------------------------------------------
# Pipeline tuning
# --------------------------------------------------------------------------
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "5000"))
GDS_GRAPH_NAME = os.getenv("GDS_GRAPH_NAME", "entity_graph")
COMMUNITY_PROPERTY = os.getenv("COMMUNITY_PROPERTY", "community_id")
BATCH_POLL_SECONDS = int(os.getenv("BATCH_POLL_SECONDS", "20"))
INDEX_NAME = os.getenv("INDEX_NAME", "graphrag")
CHUNK_VECTOR_INDEX = f"{INDEX_NAME}_chunks"
COMMUNITY_VECTOR_INDEX = f"{INDEX_NAME}_communities"
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
FORBIDDEN_CHUNK_CATEGORIES = {"Header", "Footer"}

# --------------------------------------------------------------------------
# Shared clients
# --------------------------------------------------------------------------

# One LLM client, reused for community summarization (ingest.py) AND answering (app.py)
local_llm = ChatOpenAI(
    model=LOCAL_MODEL,
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)