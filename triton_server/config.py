import os
from dotenv import load_dotenv

load_dotenv(override=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/siglip-large-patch16-384")
EMBEDDING_MODEL_D_MODEL = int(os.getenv("EMBEDDING_MODEL_D_MODEL", "768"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-m0")