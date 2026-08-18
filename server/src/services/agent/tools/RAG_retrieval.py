import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from RAG_Pipeline.retrieval.execute import retrieval_pipeline
from .base_tools import Tools

logger = logging.getLogger("agent.tools.rag_retrieval")


class RetrievalChunk(BaseModel):
    index: int
    custom_id: str
    file_name: str
    file_url: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    bbox: Optional[List[float]] = None
    type: Optional[str] = "text"
    score: Optional[float] = 0.0
    text: str = ""


class RagRetrievalOutput(BaseModel):
    query: str
    total_results: int
    chunks: List[RetrievalChunk]


@tool
async def rag_retrieval(query: str, runtime: ToolRuntime) -> RagRetrievalOutput:
    """
    Search the enterprise multimodal knowledge base for relevant text, tables, diagrams, media transcripts, and images matching the search query.

    Args:
        query: The search query or topic string to look up.

    Returns:
        Structured search results with chunk ID, file name, page number, timestamp, bounding box (bbox), document type, score, and text content.
    """
    try:
        session_id = runtime.context.get("session_id", None)
        logger.info("Executing RAG retrieval for query: %r (session_id=%s)", query, session_id)
        retrieval_result = await retrieval_pipeline.retrieve(query, session_id=session_id)
        results = retrieval_result.get("results", [])

        if not results:
            return RagRetrievalOutput(query=query, total_results=0, chunks=[])

        chunks = []
        for idx, item in enumerate(results, 1):
            file_url = item.get("file_url")
            chunk = RetrievalChunk(
                index=idx,
                custom_id=item.get("custom_id", f"doc_{idx}"),
                file_name=item.get("file_name", "unknown"),
                file_url=file_url,
                url=file_url,
                page=item.get("page"),
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
                bbox=item.get("bbox"),
                type=item.get("type", "text"),
                score=item.get("rerank_score", 0.0),
                text=item.get("chunk_text") or item.get("caption") or "",
            )
            chunks.append(chunk)

        output = RagRetrievalOutput(
            query=query,
            total_results=len(chunks),
            chunks=chunks,
        )
        return json.dumps(output.model_dump())
    except Exception as exc:
        logger.exception("Error during rag_retrieval execution: %s", exc)
        return json.dumps({"query": query, "total_results": 0, "chunks": []})


class RagRetrievalTool(Tools):
    cls_tool = rag_retrieval
