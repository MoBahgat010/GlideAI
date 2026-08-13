import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from server.src.auth.dependencies import get_current_user, get_optional_user
from server.src.models.schemas import AskRequest, AskSessionRequest
from server.src.services.agent.workflow import AgenticRAG

logger = logging.getLogger("server.routers.chat")

router = APIRouter(prefix="/api", tags=["Agentic RAG Chat"])

# Direct Singleton Instance
agent_runner = AgenticRAG()


@router.post("/sessions/{session_id}/ask")
async def ask_session(
    session_id: str,
    req: AskSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Run Agentic RAG for a specific user session.
    Loads and updates Redis working memory automatically across stateless nodes.
    Streams tokens as SSE events.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    async def event_stream():
        logger.info("Starting Agentic RAG execution for session=%s query=%r", session_id, query[:80])
        try:
            async for token in agent_runner.arun(user_message=query, session_id=session_id):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Agentic RAG streaming failed for session=%s: %s", session_id, exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ask")
async def ask_general(
    req: AskRequest,
    user: dict | None = Depends(get_optional_user),
):
    """
    Standard ask endpoint (backward compatible).
    Streams tokens as SSE with Redis working memory support.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    async def event_stream():
        logger.info("Starting general Agentic RAG execution for query=%r", query[:80])
        try:
            async for token in agent_runner.arun(user_message=query, session_id=None):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Agentic RAG streaming failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
