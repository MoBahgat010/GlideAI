import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from server.src.auth.dependencies import get_current_user, get_optional_user
from server.src.models.schemas import AskRequest, AskSessionRequest
from server.src.services.agent_service import get_agentic_rag

logger = logging.getLogger("server.routers.chat")

router = APIRouter(prefix="/api", tags=["Agentic RAG Chat"])


def _serialise_results(results: list[dict]) -> list[dict]:
    """Strip non-serialisable values from result dicts."""
    out = []
    for r in results:
        entry = {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, type(None)))}
        for nested_key in ("linked_image", "linked_caption"):
            if isinstance(r.get(nested_key), dict):
                entry[nested_key] = {
                    k: v for k, v in r[nested_key].items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
        out.append(entry)
    return out


@router.post("/sessions/{session_id}/ask")
async def ask_session(
    session_id: str,
    req: AskSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Run Agentic RAG for a specific user session.
    Loads and updates Redis working memory automatically.
    Streams tokens as SSE events.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    pipeline, generator = get_agentic_rag()

    async def event_stream():
        logger.info("Starting Agentic RAG retrieval for session=%s query=%r", session_id, query[:80])
        try:
            retrieval_result = await pipeline.retrieve(query)
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        results = retrieval_result.get("results", [])
        logger.info("Retrieval returned %d results; starting Agentic answer stream", len(results))

        # Stream Agentic RAG LLM tokens with Redis working memory
        try:
            async for token in generator.stream(query, results, session_id=session_id):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Agentic RAG streaming failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        # Final SSE event
        safe_results = _serialise_results(results)
        yield f"data: {json.dumps({'type': 'done', 'results': safe_results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ask")
async def ask_general(
    req: AskRequest,
    user: dict | None = Depends(get_optional_user),
):
    """
    Standard ask endpoint (backward compatible).
    Streams tokens as Server-Sent Events.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    pipeline, generator = get_agentic_rag()

    async def event_stream():
        logger.info("Starting general RAG retrieval for query=%r", query[:80])
        try:
            retrieval_result = await pipeline.retrieve(query)
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        results = retrieval_result.get("results", [])
        logger.info("Retrieval returned %d results; starting answer stream", len(results))

        try:
            async for token in generator.stream(query, results, session_id=None):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Answer streaming failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        safe_results = _serialise_results(results)
        yield f"data: {json.dumps({'type': 'done', 'results': safe_results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
