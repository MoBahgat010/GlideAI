from datetime import datetime
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from server.src.auth.dependencies import get_current_user
from server.src.db.mongo import get_database
from server.src.db.redis import get_redis_client
from server.src.models.schemas import SessionCreate, SessionEndResponse, SessionResponse
from tasks import extract_session_memory

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Create a new user-isolated RAG session."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    user_id = str(current_user["_id"])

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "username": current_user["username"],
        "title": payload.title,
        "status": "active",
        "created_at": datetime.utcnow(),
        "memory_extracted": False,
    }
    await db.sessions.insert_one(session_doc)

    return SessionResponse(
        session_id=session_id,
        user_id=user_id,
        title=payload.title,
        status="active",
        created_at=session_doc["created_at"],
        memory_extracted=False,
    )


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List all sessions belonging to the authenticated user."""
    user_id = str(current_user["_id"])
    cursor = db.sessions.find({"user_id": user_id}).sort("created_at", -1)
    sessions = []
    async for s in cursor:
        sessions.append(
            SessionResponse(
                session_id=s["session_id"],
                user_id=s["user_id"],
                title=s.get("title", "RAG Session"),
                status=s.get("status", "active"),
                created_at=s.get("created_at", datetime.utcnow()),
                memory_extracted=s.get("memory_extracted", False),
            )
        )
    return sessions


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get details and Redis working memory transcript for a specific session."""
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(404, "Session not found or access denied.")

    # Fetch working history from Redis
    r_client = await get_redis_client()
    key = f"session:{session_id}:working_memory"
    import json
    raw = await r_client.get(key)
    history = json.loads(raw) if raw else []

    return {
        "session_id": session["session_id"],
        "title": session.get("title", "RAG Session"),
        "status": session.get("status", "active"),
        "created_at": session.get("created_at"),
        "memory_extracted": session.get("memory_extracted", False),
        "history": history,
    }


@router.post("/{session_id}/end", response_model=SessionEndResponse)
async def end_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Close/end a session and dispatch Celery worker to extract
    episodic and semantic memory from the conversation transcript.
    """
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(404, "Session not found or access denied.")

    # Update session status
    await db.sessions.update_one(
        {"session_id": session_id},
        {"$set": {"status": "ending", "ended_at": datetime.utcnow()}},
    )

    # Dispatch Celery background memory extraction job
    task = extract_session_memory.delay(session_id=session_id, user_id=user_id)

    return SessionEndResponse(
        session_id=session_id,
        status="ending",
        task_id=task.id,
        message="Session closed. Celery memory extraction worker dispatched.",
    )


@router.get("/{session_id}/memories")
async def get_session_memories(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Retrieve extracted episodic and semantic memories for a session."""
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(404, "Session not found or access denied.")

    episodic = await db.episodic_memories.find_one({"session_id": session_id})
    semantic = await db.semantic_memories.find_one({"session_id": session_id})

    # Clean Mongo IDs
    if episodic:
        episodic["_id"] = str(episodic["_id"])
    if semantic:
        semantic["_id"] = str(semantic["_id"])

    return {
        "session_id": session_id,
        "episodic_memory": episodic or {},
        "semantic_memory": semantic or {},
    }
