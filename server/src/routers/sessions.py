from langchain_core.messages import SystemMessage
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from langchain_core.messages import HumanMessage
from config import (
    ENABLE_SEMANTIC_MEMORY,
    ENABLE_EPISODIC_MEMORY,
    UPLOAD_DIR,
)

from src.auth.dependencies import get_current_user
from src.db.mongo import get_database
from src.db.redis import RedisSessionUtils
from src.models.schemas import (
    SessionCreate,
    SessionEndResponse,
    SessionResponse,
    GenerateTitleRequest,
    FileMetadata,
)
from src.jobs.tasks import extract_session_memory
from src.services.agent.models.summarizer import summarizer

logger = logging.getLogger("server.routers.sessions")
router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


def _extract_session_files(session_doc: dict, session_id: str) -> list[dict]:
    """Extract session files list with Cloudinary file_url directly from MongoDB."""
    files_map = {}
    for f in session_doc.get("files", []):
        fn = f.get("filename")
        if fn:
            f_url = f.get("file_url")
            files_map[fn] = {
                "filename": fn,
                "size": f.get("size", 0),
                "file_type": f.get("file_type", "document"),
                "file_url": f_url,
                "url": f_url,
                "uploaded_at": f.get("uploaded_at"),
            }

    return list(files_map.values())


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    session_id = uuid.uuid4().hex[:12]
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
        "files": [],
    }
    await db.sessions.insert_one(session_doc)
    logger.info("Created new session %s for user %s", session_id, current_user["username"])

    session_dir = Path(UPLOAD_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    return SessionResponse(
        session_id=session_id,
        user_id=user_id,
        title=session_doc["title"],
        status="active",
        created_at=session_doc["created_at"],
        memory_extracted=False,
        files=[],
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
        files = _extract_session_files(s, s["session_id"])
        sessions.append(
            SessionResponse(
                session_id=s["session_id"],
                user_id=s["user_id"],
                title=s.get("title", "RAG Session"),
                status=s.get("status", "active"),
                created_at=s.get("created_at", datetime.now(timezone.utc)),
                memory_extracted=s.get("memory_extracted", False),
                files=files,
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

    history_client = RedisSessionUtils.get_chat_history(session_id)

    redis_messages = history_client.messages

    formatted_messages = []

    if len(redis_messages) == 0:
        raw_messages = session.get("messages") or session.get("history") or []
        for m in raw_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            ts = m.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.isoformat()

            formatted_messages.append({
                "role": role,
                "content": content,
                "citations": m.get("citations", []),
                "timestamp": ts,
            })

            if role == "user":
                history_client.add_user_message(content)
            elif role in ("assistant", "ai"):
                history_client.add_ai_message(content)

        if formatted_messages:
            logger.info(
                "Cold start hydration: seeded %d messages from MongoDB into Redis for session %s",
                len(formatted_messages),
                session_id,
            )
    else:
        for msg in redis_messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            formatted_messages.append({
                "role": role,
                "content": msg.content,
            })

    files = _extract_session_files(session, session_id)

    created_at = session.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    return {
        "id": session["session_id"],
        "session_id": session["session_id"],
        "title": session.get("title", "RAG Session"),
        "status": session.get("status", "active"),
        "created_at": created_at,
        "memory_extracted": session.get("memory_extracted", False),
        "files": files,
        "history": formatted_messages,
    }


@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    await db.sessions.delete_one({"session_id": session_id})
    await db.episodic_memories.delete_many({"session_id": session_id})
    await db.semantic_memories.delete_many({"session_id": session_id})

    await RedisSessionUtils.delete_session_memory(session_id)

    session_dir = Path(UPLOAD_DIR) / session_id
    if session_dir.exists():
        await asyncio.to_thread(shutil.rmtree, str(session_dir), True)

    logger.info("Deleted session %s for user %s", session_id, current_user["username"])
    return {"status": "deleted", "session_id": session_id}


@router.get("/{session_id}/files")
async def list_session_files(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    files = _extract_session_files(session, session_id)
    return {"session_id": session_id, "files": files}


@router.delete("/{session_id}/files/{filename}")
async def delete_session_file(
    session_id: str,
    filename: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    await db.sessions.update_one(
        {"session_id": session_id},
        {"$pull": {"files": {"filename": filename}}},
    )

    file_path = Path(UPLOAD_DIR) / session_id / filename
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning("Failed to delete physical file %s: %s", file_path, e)

    return {"status": "deleted", "filename": filename, "session_id": session_id}


@router.post("/{session_id}/generate-title")
async def generate_session_title(
    session_id: str,
    req: GenerateTitleRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Generate concise title using summarizer model and stream tokens via SSE, then update MongoDB."""
    user_id = str(current_user["_id"])
    session = await db.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    prompt_text = req.prompt.strip()
    if not prompt_text:
        raise HTTPException(400, "Prompt must not be empty.")

    system_inst = (
        "You are an expert title generator for conversation sessions in an enterprise multi-modal AI system. "
        "Generate a brief, clear, descriptive title (strictly 3 to 6 words maximum) for a conversation starting with the user's initial prompt. "
        "Return ONLY the plain title text with NO quotation marks, NO markdown, and NO trailing punctuation."
    )

    async def title_stream():
        logger.info("Generating session title for session_id=%s from prompt: %r", session_id, prompt_text[:60])
        messages = [
            SystemMessage(content=system_inst),
            HumanMessage(content=f"Initial Prompt: {prompt_text}"),
        ]

        title_parts = []
        try:
            async for chunk in summarizer.astream(messages):
                if chunk and chunk.content:
                    token = str(chunk.content)
                    title_parts.append(token)
                    yield f"data: {json.dumps({'type': 'title_token', 'content': token})}\n\n"

            final_title = "".join(title_parts).strip().replace('"', '').replace("'", "").strip()
            if final_title.lower().startswith("title:"):
                final_title = final_title[6:].strip()
            if not final_title:
                final_title = prompt_text[:30] + ("..." if len(prompt_text) > 30 else "")

            await db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"title": final_title, "updated_at": datetime.now(timezone.utc)}},
            )
            logger.info("Updated title for session %s to %r", session_id, final_title)
            yield f"data: {json.dumps({'type': 'title_done', 'title': final_title})}\n\n"

        except Exception as exc:
            logger.exception("Error generating title for session %s: %s", session_id, exc)
            fallback_title = prompt_text[:35]
            await db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"title": fallback_title, "updated_at": datetime.now(timezone.utc)}},
            )
            yield f"data: {json.dumps({'type': 'title_done', 'title': fallback_title})}\n\n"

    return StreamingResponse(title_stream(), media_type="text/event-stream")


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
        },
        "episodic_memory": episodic or {},
        "semantic_memory": semantic or {},
    }
