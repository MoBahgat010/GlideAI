import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from src.auth.dependencies import get_current_user
from src.db.mongo import get_database
from src.models.schemas import AskSessionRequest
from src.services.agent.workflow import agent_runner

logger = logging.getLogger("server.routers.chat")
router = APIRouter(prefix="/api/sessions", tags=["Agentic RAG Chat"])

@router.post("/{session_id}/ask")
async def ask_session(
    session_id: str,
    req: AskSessionRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    user_id = str(current_user["_id"])

    async def event_stream():
        logger.info("Starting Agentic RAG streaming for user=%s (id=%s) session=%s", current_user.get("username"), user_id, session_id)
        try:
            async for event in agent_runner.astream_response(
                user_message=query,
                session_id=session_id,
                user_id=user_id,
                db=db,
            ):
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Error during chat stream for session %s: %s", session_id, exc)
            err_msg = f"Agent execution error: {str(exc)}"
            yield f"data: {json.dumps({'type': 'error', 'content': err_msg})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
