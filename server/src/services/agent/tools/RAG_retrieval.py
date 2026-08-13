import logging
from langchain_core.tools import tool
from RAG_Pipeline.retrieval.execute import get_retrieval_pipeline
from server.src.services.agent.tools.base_tools import Tools

logger = logging.getLogger("agent.tools.rag_retrieval")

@tool
async def rag_retrieval(query: str) -> str:
    """
    Search the enterprise multimodal knowledge base for relevant text, tables, and images matching the search query.

    Args:
        query: The search query or topic string to look up.

    Returns:
        Structured search results formatted with chunk ID, file name, page number, bounding box (bbox), document type, score, and content.
    """
    try:
        logger.info("Executing RAG retrieval for query: %r", query)
        retrieval_result = await get_retrieval_pipeline().retrieve(query)
        results = retrieval_result.get("results", [])

        if not results:
            return "No matching documents found in the enterprise knowledge base."

        output_snippets = []
        for idx, item in enumerate(results, 1):
            chunk_id = item.get("custom_id", f"doc_{idx}")
            file_name = item.get("file_name", "N/A")
            page = item.get("page", "N/A")
            bbox = item.get("bbox", "N/A")
            doc_type = item.get("type", "text")
            score = item.get("rerank_score", 0.0)
            text = item.get("chunk_text") or item.get("caption") or ""

            snippet = (
                f"--- Result [{idx}] ---\n"
                f"Chunk ID: {chunk_id}\n"
                f"File Name: {file_name}\n"
                f"Page: {page}\n"
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
