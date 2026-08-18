import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.auth.dependencies import get_current_user
from src.models.schemas import HiTLResumeRequest

logger = logging.getLogger("server.interrupts.router")
router = APIRouter(prefix="/api/sessions", tags=["HiTL Approval"])


def _get_agent_runner():
    from src.services.agent.workflow import agent_runner
    return agent_runner


@router.post("/{session_id}/approve")
async def approve_hitl(
    session_id: str,
    req: HiTLResumeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Resume a paused agent after a HiTL interrupt decision."""
    agent_runner = _get_agent_runner()
    decisions = [d.model_dump(exclude_none=True) for d in req.decisions]

    user_id = str(current_user["_id"])

    async def event_stream():
        logger.info(
            "Resuming HiTL for user=%s (id=%s) session=%s decisions=%s",
            current_user.get("username"),
            user_id,
            session_id,
            decisions,
        )
        try:
            async for token in agent_runner.resume(decisions, session_id, user_id=user_id):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            logger.exception("Error resuming HiTL for session %s: %s", session_id, exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
