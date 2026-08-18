import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from langchain_core.messages import SystemMessage, HumanMessage

from config import UPLOAD_DIR
from src.auth.dependencies import get_current_user
from src.db.mongo import mongo
from src.models.schemas import (
    SessionCreate,
    SessionEndResponse,
    SessionResponse,
    SessionTitle,
    GenerateTitleRequest,
)
from src.services.agent.models.summarizer import summarizer

logger = logging.getLogger("server.routers.sessions")
router = APIRouter(prefix="/api/sessions", tags=["Sessions"])
title_generator = summarizer.with_structured_output(SessionTitle)


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
        "files": [],
        "messages": [],
    }
    mongo.sessions.insert_one(session_doc)
    logger.info("Created new session %s for user %s", session_id, current_user["username"])

    session_dir = Path(UPLOAD_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    return SessionResponse(
        session_id=session_id,
        user_id=user_id,
        title=session_doc["title"],
        status="active",
        created_at=session_doc["created_at"],
        files=[],
    )


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    cursor = mongo.sessions.find({"user_id": user_id}).sort("created_at", -1)
    sessions = []
    for s in cursor:
        files = _extract_session_files(s, s["session_id"])
        sessions.append(
            SessionResponse(
                session_id=s["session_id"],
                user_id=s["user_id"],
                title=s.get("title", "RAG Session"),
                status=s.get("status", "active"),
                created_at=s.get("created_at", datetime.now(timezone.utc)),
                files=files,
            )
        )
    return sessions


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    session = mongo.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    raw_messages = session.get("messages") or session.get("history") or []
    formatted_messages = []
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
        "files": files,
        "history": formatted_messages,
    }


@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    session = mongo.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    mongo.sessions.delete_one({"session_id": session_id})

    session_dir = Path(UPLOAD_DIR) / session_id
    if session_dir.exists():
        await asyncio.to_thread(shutil.rmtree, str(session_dir), True)

    logger.info("Deleted session %s for user %s", session_id, current_user["username"])
    return {"status": "deleted", "session_id": session_id}


@router.get("/{session_id}/files")
async def list_session_files(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    session = mongo.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    files = _extract_session_files(session, session_id)
    return {"session_id": session_id, "files": files}


@router.get("/{session_id}/files/{filename}")
async def get_session_file(
    session_id: str,
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """Serve or redirect to a session document/media file for viewing."""
    user_id = str(current_user["_id"])
    session = mongo.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    file_path = Path(UPLOAD_DIR) / session_id / filename
    if file_path.exists() and file_path.is_file():
        media_type = "application/pdf" if filename.lower().endswith(".pdf") else None
        return FileResponse(file_path, filename=filename, media_type=media_type)

    # If not on local disk, check if Cloudinary URL is stored in session
    for f in session.get("files", []):
        if f.get("filename") == filename and (f.get("file_url") or f.get("url")):
            return RedirectResponse(url=f.get("file_url") or f.get("url"))

    raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found.")


@router.delete("/{session_id}/files/{filename}")
async def delete_session_file(
    session_id: str,
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    session = mongo.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    mongo.sessions.update_one(
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
):
    """Generate concise title using structured decoding, then update MongoDB."""
    user_id = str(current_user["_id"])
    session = mongo.sessions.find_one({"session_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or access denied.")

    prompt_text = req.prompt.strip()
    if not prompt_text:
        raise HTTPException(400, "Prompt must not be empty.")

    async def title_stream():
        logger.info("Generating structured session title for session_id=%s", session_id)
        try:
            result: SessionTitle = await title_generator.ainvoke([
                SystemMessage(content="You are an expert title generator for conversation sessions in an enterprise multi-modal AI system. Generate a concise, descriptive title (strictly 3 to 6 words maximum)."),
                HumanMessage(content=f"Initial Prompt: {prompt_text}"),
            ])
            final_title = result.title.strip()
            mongo.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"title": final_title, "updated_at": datetime.now(timezone.utc)}},
            )
            logger.info("Updated title for session %s to %r", session_id, final_title)
            yield f"data: {json.dumps({'type': 'title_done', 'title': final_title})}\n\n"

        except Exception as exc:
            logger.exception("Error generating structured title for session %s: %s", session_id, exc)
            fallback_title = "New Session"
            mongo.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"title": fallback_title, "updated_at": datetime.now(timezone.utc)}},
            )
            yield f"data: {json.dumps({'type': 'title_done', 'title': fallback_title})}\n\n"

    return StreamingResponse(title_stream(), media_type="text/event-stream")