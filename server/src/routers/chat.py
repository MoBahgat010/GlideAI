import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.auth.dependencies import get_current_user
from src.models.schemas import AskSessionRequest
from src.services.agent.workflow import AgenticRAG

logger = logging.getLogger("server.routers.chat")
router = APIRouter(prefix="/api/sessions", tags=["Agentic RAG Chat"])

agent_runner = AgenticRAG()

@router.post("/{session_id}/ask")
async def ask_session(
    session_id: str,
    req: AskSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    async def event_stream():
        logger.info("Starting Agentic RAG streaming for user=%s session=%s", current_user.get("username"), session_id)
        async for event in agent_runner.astream_response(user_message=query, session_id=session_id):
            payload = json.dumps(event)
            yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
