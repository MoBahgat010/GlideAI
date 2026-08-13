from functools import lru_cache

from RAG_Pipeline.retrieval.pipeline import RetrievalPipeline
from RAG_Pipeline.retrieval.reranker import HybridReranker
from RAG_Pipeline.ingestion.embedding import MultimodalEncoder
from RAG_Pipeline.storage.weaviate import WeaviateVDB
from RAG_Pipeline.storage.vector_database import VDB
from config import TRITON_GRPC_URL, WEAVIATE_API_KEY, WEAVIATE_REST_ENDPOINT, INDEX_NAME, RETRIEVE_TOP_K, RERANK_TOP_K


@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    """Lazy-initialised singleton — built on first call, not at import time."""
    encoder = MultimodalEncoder(url=TRITON_GRPC_URL)
    reranker = HybridReranker(url=TRITON_GRPC_URL)
    weaviate = WeaviateVDB(
        endpoint=WEAVIATE_REST_ENDPOINT,
        api_key=WEAVIATE_API_KEY,
        index=INDEX_NAME,
        dimension=encoder.d_model,
    )
    vdb = VDB(weaviate)
    return RetrievalPipeline(
        encoder=encoder,
        reranker=reranker,
        vdb=vdb,
        retrieve_top_k=RETRIEVE_TOP_K,
        rerank_top_k=RERANK_TOP_K,
    )


# Backwards-compatible name — evaluates lazily on first access
retrieval_pipeline = None  # resolved via get_retrieval_pipeline() at call site
