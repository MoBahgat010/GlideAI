import asyncio
import logging
from pathlib import Path
import shutil
from typing import Optional

import aiofiles
import aiofiles.os
from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from config import CHUNKS_DIR, UPLOAD_DIR

from src.auth.dependencies import get_optional_user
from src.jobs.tasks import celery_app, run_ingestion

logger = logging.getLogger("server.routers.ingest")
router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

@router.post("/chunk")
async def receive_chunk(
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """Receive a single file slice and store it temporarily on disk."""
    slot_dir = Path(CHUNKS_DIR) / upload_id
    await aiofiles.os.makedirs(str(slot_dir), exist_ok=True)

    chunk_path = slot_dir / f"{chunk_index:05d}.part"
    content = await file.read()

    async with aiofiles.open(str(chunk_path), "wb") as f:
        await f.write(content)

    logger.info("Received chunk %d/%d for upload_id=%s file=%s session=%s (%d bytes)", chunk_index + 1, total_chunks, upload_id, filename, session_id, len(content))
    return {"upload_id": upload_id, "chunk_index": chunk_index, "received": True}


@router.post("/finalize")
async def finalize_upload(
    upload_id: str = Form(...),
    filename: str = Form(...),
    total_chunks: int = Form(...),
    session_id: Optional[str] = Form(None),
    user: Optional[dict] = Depends(get_optional_user),
):
    slot_dir = Path(CHUNKS_DIR) / upload_id
    session_upload_dir = Path(UPLOAD_DIR) / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    final_path = session_upload_dir / filename

    logger.info("Finalizing upload_id=%s file=%s in session folder %s", upload_id, filename, session_upload_dir)

    async with aiofiles.open(str(final_path), "wb") as out_f:
        for i in range(total_chunks):
            part = slot_dir / f"{i:05d}.part"
            if not part.exists():
                raise HTTPException(400, f"Missing chunk slice {i} for upload {upload_id}")
            async with aiofiles.open(str(part), "rb") as in_f:
                await out_f.write(await in_f.read())

    await asyncio.to_thread(shutil.rmtree, str(slot_dir), True)
    logger.info("Assembled %s (%d bytes) in session folder %s", final_path.name, final_path.stat().st_size, session_upload_dir)

    # Ingest the session folder to process all uploaded PDF and media files in one pipeline run
    storage_key = str(session_upload_dir)
    job_id = upload_id

    task = run_ingestion.delay(job_id=job_id, storage_keys=storage_key, session_id=session_id)
    logger.info("Dispatched Celery ingestion task %s for session=%s", task.id, session_id)

    return {"task_id": task.id, "job_id": job_id, "file": filename, "session_id": session_id}


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    state = result.state

    if state == "PENDING":
        return {"state": state, "stage": "PENDING", "message": "Job queued in broker…", "pct": 0.0}

    if state == "PROGRESS":
        meta = result.info or {}
        return {
            "state": state,
            "stage": meta.get("stage", "PROCESSING"),
            "message": meta.get("message", "Processing…"),
            "pct": meta.get("pct", 0.3),
        }

    if state == "SUCCESS":
        return {"state": state, "stage": "DONE", "message": "Ingestion complete and indexed.", "pct": 1.0, "result": result.result}

    if state == "FAILURE":
        return {"state": state, "stage": "FAILED", "message": str(result.info), "pct": 0.0}

    return {"state": state, "stage": state, "message": "", "pct": 0.0}
