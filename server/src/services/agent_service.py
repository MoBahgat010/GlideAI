import logging
from openai import AsyncOpenAI
from config import (
    INDEX_NAME,
    QWEN_MODEL,
    QWEN_SERVER_URL,
    REDIS_URL,
    RERANK_TOP_K,
    RETRIEVE_TOP_K,
    WEAVIATE_API_KEY,
    WEAVIATE_REST_ENDPOINT,
    HF_TOKEN,
    TRITON_URL,
)

from RAG_Pipeline.ingestion.embedding import MultimodalEncoder
from RAG_Pipeline.retrieval.answer import AgenticAnswerGenerator
from RAG_Pipeline.retrieval.pipeline import RetrievalPipeline
from RAG_Pipeline.retrieval.reranker import HybridReranker
from RAG_Pipeline.storage.weaviate import WeaviateVDB
from huggingface_hub import login

try:
    if HF_TOKEN:
        login(HF_TOKEN)
except Exception as e:
    logging.warning("HuggingFace login failed or skipped: %s", e)

logger = logging.getLogger("server.services.agent")

_retrieval_pipeline: RetrievalPipeline | None = None
_answer_generator: AgenticAnswerGenerator | None = None


def get_agentic_rag() -> tuple[RetrievalPipeline, AgenticAnswerGenerator]:
    """Retrieve or initialize the RAG pipeline and Agentic RAG answer generator."""
    global _retrieval_pipeline, _answer_generator
    if _retrieval_pipeline is None or _answer_generator is None:
        logger.info("Initializing Retrieval Pipeline and Agentic RAG Answer Generator via Triton Server (%s)...", TRITON_URL)
        
        reranker = HybridReranker(url=TRITON_URL)
        encoder = MultimodalEncoder(url=TRITON_URL)

        vdb = WeaviateVDB(
            endpoint=WEAVIATE_REST_ENDPOINT,
            api_key=WEAVIATE_API_KEY,
            index=INDEX_NAME,
            dimension=encoder.d_model,
        )

        qwen_base_url = QWEN_SERVER_URL.rstrip("/")
        if not qwen_base_url.endswith("/v1"):
            qwen_base_url += "/v1"

        logger.info("Connecting AsyncOpenAI client to URL: %s", qwen_base_url)
        client = AsyncOpenAI(api_key="EMPTY", base_url=qwen_base_url)
        model_name = QWEN_MODEL

        _retrieval_pipeline = RetrievalPipeline(
            encoder=encoder,
            vdb=vdb,
            local_client=client,
            local_model=model_name,
            retrieve_top_k=RETRIEVE_TOP_K,
            rerank_top_k=RERANK_TOP_K,
            reranker=reranker,
        )
        _answer_generator = AgenticAnswerGenerator(
            client=client,
            model=model_name,
            redis_url=REDIS_URL,
        )
    return _retrieval_pipeline, _answer_generator
