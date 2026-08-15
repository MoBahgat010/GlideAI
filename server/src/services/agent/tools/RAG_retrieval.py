import logging
from langchain_core.tools import tool

from RAG_Pipeline.retrieval.execute import retrieval_pipeline
from .base_tools import Tools

logger = logging.getLogger("agent.tools.rag_retrieval")

@tool
async def rag_retrieval(query: str) -> str:
    """
    Search the enterprise multimodal knowledge base for relevant text, tables, diagrams, media transcripts, and images matching the search query.

    Args:
        query: The search query or topic string to look up.

    Returns:
        Structured search results with chunk ID, file name, page number, timestamp, bounding box (bbox), document type, score, and text content.
    """
    try:
        logger.info("Executing RAG retrieval for query: %r", query)
        retrieval_result = await retrieval_pipeline.retrieve(query)
        results = retrieval_result.get("results", [])

        if not results:
            return "No matching documents found in the enterprise knowledge base."

        output_snippets = []
        for idx, item in enumerate(results, 1):
            chunk_id = item.get("custom_id", f"doc_{idx}")
            file_name = item.get("file_name", "unknown")
            page = item.get("page", 1)
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            bbox = item.get("bbox")
            doc_type = item.get("type", "text")
            score = item.get("rerank_score", 0.0)
            text = item.get("chunk_text") or item.get("caption") or ""

            pos_info = f"Page: {page}" if doc_type != "transcript" else f"Time: {start_time:.1f}s - {end_time:.1f}s"

            snippet = (
                f"--- Result {idx} ---\n"
                f"Chunk ID: {chunk_id}\n"
                f"File Name: {file_name}\n"
                f"{pos_info}\n"
                f"BBox: {bbox}\n"
                f"Type: {doc_type}\n"
                f"Rerank Score: {score:.4f}\n"
                f"Content:\n{text}"
            )
            output_snippets.append(snippet)

        return "\n\n".join(output_snippets)
    except Exception as exc:
        logger.exception("Error during rag_retrieval execution: %s", exc)
        return f"Error executing RAG retrieval: {str(exc)}"


class RagRetrievalTool(Tools):
    cls_tool = rag_retrieval
