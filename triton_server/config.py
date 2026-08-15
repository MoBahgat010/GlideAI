import os
from dotenv import load_dotenv

load_dotenv(override=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL_D_MODEL = int(os.getenv("EMBEDDING_MODEL_D_MODEL"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")