from datetime import datetime, timezone
import json
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import ENABLE_SEMANTIC_MEMORY, ENABLE_EPISODIC_MEMORY, ENABLE_WORKING_MEMORY

from src.auth.dependencies import get_current_user
from src.db.mongo import get_database
from src.db.redis import redis_client
from src.models.schemas import SessionCreate, SessionEndResponse, SessionResponse
from src.jobs.tasks import extract_session_memory

logger = logging.getLogger("server.routers.sessions")
router = APIRouter(prefix="/api/sessions", tags=["Sessions"])

@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    user_id = str(current_user["_id"])
    now = datetime.now(timezone.utc)

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "username": current_user["username"],
        "title": payload.title.strip() or "New RAG Session",
        "status": "active",
        "created_at": now,
        "memory_extracted": False,
    }
    await db.sessions.insert_one(session_doc)
    logger.info("Created new session %s for user %s", session_id, current_user["username"])

    return SessionResponse(
        session_id=session_id,
        user_id=user_id,
        title=session_doc["title"],
        status="active",
        created_at=session_doc["created_at"],
        memory_extracted=False,
    )


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
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
                created_at=s.get("created_at", datetime.now(timezone.utc)),
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
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    history = []
    if ENABLE_WORKING_MEMORY:
        try:
            key = f"session:{session_id}:working_memory"
            raw = await redis_client.get(key)
            if raw:
                history = json.loads(raw)
        except Exception as exc:
            logger.warning("Failed to load working memory for session %s: %s", session_id, exc)

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
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    now = datetime.now(timezone.utc)
    await db.sessions.update_one(
        {"session_id": session_id},
        {"$set": {"status": "ending", "ended_at": now}},
    )

    if ENABLE_SEMANTIC_MEMORY or ENABLE_EPISODIC_MEMORY:
        task = extract_session_memory.delay(session_id=session_id, user_id=user_id)
        task_id = task.id
        msg = "Session closed. Celery memory extraction worker dispatched."
    else:
        task_id = "skipped_feature_flag_disabled"
        msg = "Session closed. Memory extraction skipped as feature flags are disabled."
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "ended", "ended_at": now}},
        )

    return SessionEndResponse(
        session_id=session_id,
        status="ending",
        task_id=task_id,
        message=msg,
    )


@router.get("/{session_id}/memories")
async def get_session_memories(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    episodic = await db.episodic_memories.find_one({"session_id": session_id})
    semantic = await db.semantic_memories.find_one({"session_id": session_id})

    if episodic:
        episodic["_id"] = str(episodic["_id"])
    if semantic:
        semantic["_id"] = str(semantic["_id"])

    return {
        "session_id": session_id,
        "feature_flags": {
            "enable_semantic_memory": ENABLE_SEMANTIC_MEMORY,
            "enable_episodic_memory": ENABLE_EPISODIC_MEMORY,
            "enable_working_memory": ENABLE_WORKING_MEMORY,
        },
        "episodic_memory": episodic or {},
        "semantic_memory": semantic or {},
    }
