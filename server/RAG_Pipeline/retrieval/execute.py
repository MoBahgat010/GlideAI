from .pipeline import RetrievalPipeline
from .reranker import HybridReranker
from ..ingestion.embedding import MultimodalEncoder
from ..storage.weaviate import WeaviateVDB

from config import (
    TRITON_GRPC_URL,
    WEAVIATE_API_KEY,
    WEAVIATE_REST_ENDPOINT,
    INDEX_NAME,
    RETRIEVE_TOP_K,
    RERANK_TOP_K,
    EMBEDDING_MODEL_D_MODEL,
)

encoder = MultimodalEncoder(url=TRITON_GRPC_URL, d_model=int(EMBEDDING_MODEL_D_MODEL))
reranker = HybridReranker(url=TRITON_GRPC_URL)

vdb = WeaviateVDB(
    endpoint=WEAVIATE_REST_ENDPOINT,
    api_key=WEAVIATE_API_KEY,
    index=INDEX_NAME,
    dimension=encoder.d_model,
)

retrieval_pipeline = RetrievalPipeline(
    encoder=encoder,
    reranker=reranker,
    vdb=vdb,
    retrieve_top_k=RETRIEVE_TOP_K,
    rerank_top_k=RERANK_TOP_K,
)
