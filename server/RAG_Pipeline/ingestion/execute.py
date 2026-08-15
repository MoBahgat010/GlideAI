from .pipeline import IngestionPipeline
from .loader import PDFLoader
from .embedding import MultimodalEncoder
from .chunking import SemanticChunker
from ..storage.vector_database import VDB
from ..storage.weaviate import WeaviateVDB

from config import (
    WEAVIATE_API_KEY,
    WEAVIATE_REST_ENDPOINT,
    INDEX_NAME,
    TRITON_GRPC_URL,
    MAX_CHARS,
    OVERLAP_CHARS,
    EMBEDDING_MODEL_D_MODEL,
    EMBED_BATCH,
)


loader = PDFLoader()
chunker = SemanticChunker(max_chars=MAX_CHARS, overlap_chars=OVERLAP_CHARS)
encoder = MultimodalEncoder(url=TRITON_GRPC_URL, d_model=int(EMBEDDING_MODEL_D_MODEL))

vdb = VDB(
    strategy=WeaviateVDB(
        endpoint=WEAVIATE_REST_ENDPOINT,
        api_key=WEAVIATE_API_KEY,
        index=INDEX_NAME,
        dimension=encoder.d_model,
    )
)

ingestion_pipeline = IngestionPipeline(
    loader=loader,
    chunker=chunker,
    encoder=encoder,
    vdb=vdb,
    batch_size=EMBED_BATCH,
)
