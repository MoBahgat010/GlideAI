import logging
from langchain_core.tools import tool
from ..models.summarizer import summarizer
from .base_tools import Tools

logger = logging.getLogger("agent.tools.document_summarizer")

@tool
async def document_summarizer(text: str) -> str:
    """
    Summarize long document passages or retrieved search results into concise key points.

    Args:
        text: The text content to summarize.

    Returns:
        A concise summary highlighting the primary insights.
    """
    try:
        logger.info("Executing document_summarizer on text snippet of length %d", len(text))
        prompt = f"Summarize the following text into clear bullet points:\n\n{text}"
        response = await summarizer.ainvoke(prompt)
        return response.content
    except Exception as exc:
        logger.exception("Error executing document_summarizer: %s", exc)
        return f"Failed to generate summary: {str(exc)}"


class DocumentSummarizerTool(Tools):
    cls_tool = document_summarizer
