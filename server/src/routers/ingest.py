import asyncio
import logging
from pathlib import Path
import shutil

import aiofiles
import aiofiles.os
from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends

from config import CHUNKS_DIR, UPLOAD_DIR

from server.src.jobs.tasks import celery_app, run_ingestion

logger = logging.getLogger("server.routers.ingest")

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])


@router.post("/chunk")
async def receive_chunk(
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
):
    """Receive a single file slice and store it asynchronously."""
    slot_dir = Path(CHUNKS_DIR) / upload_id
    await aiofiles.os.makedirs(str(slot_dir), exist_ok=True)

    chunk_path = slot_dir / f"{chunk_index:05d}.part"
    content = await file.read()

    async with aiofiles.open(str(chunk_path), "wb") as f:
        await f.write(content)

    logger.info(
        "Received chunk %d/%d for upload_id=%s file=%s bytes=%d",
        chunk_index + 1, total_chunks, upload_id, filename, len(content),
    )
    return {"upload_id": upload_id, "chunk_index": chunk_index, "received": True}


@router.post("/finalize")
async def finalize_upload(
    upload_id: str = Form(...),
    filename: str = Form(...),
    total_chunks: int = Form(...),
):
    """Reassemble slices into final document and trigger Celery ingestion task."""
    slot_dir = Path(CHUNKS_DIR) / upload_id
    upload_dir_path = Path(UPLOAD_DIR)
    upload_dir_path.mkdir(parents=True, exist_ok=True)
    final_path = upload_dir_path / filename

    logger.info(
        "Finalising upload_id=%s filename=%s total_chunks=%d",
        upload_id, filename, total_chunks,
    )

    async with aiofiles.open(str(final_path), "wb") as out_f:
        for i in range(total_chunks):
            part = slot_dir / f"{i:05d}.part"
            if not part.exists():
                raise HTTPException(400, f"Missing chunk {i} for upload {upload_id}")
            async with aiofiles.open(str(part), "rb") as in_f:
                await out_f.write(await in_f.read())

    # Remove temp chunk directory
    await asyncio.to_thread(shutil.rmtree, str(slot_dir), True)
    logger.info("Assembled %s (%d bytes)", final_path, final_path.stat().st_size)

    storage_key = str(final_path)
    job_id = upload_id

    task = run_ingestion.delay(job_id=job_id, storage_keys=storage_key)
    logger.info("Dispatched Celery ingestion task %s for job %s", task.id, job_id)

    return {"task_id": task.id, "job_id": job_id, "file": filename}


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    """Poll Celery task status for ingestion progress."""
    result = AsyncResult(task_id, app=celery_app)
    state = result.state

    if state == "PENDING":
        return {"state": state, "stage": "PENDING", "message": "Job queued…", "pct": 0.0}

    if state == "PROGRESS":
        meta = result.info or {}
        return {
            "state": state,
            "stage": meta.get("stage", ""),
            "message": meta.get("message", ""),
            "pct": meta.get("pct", 0.0),
        }

    if state == "SUCCESS":
        return {"state": state, "stage": "DONE", "message": "Ingestion complete.", "pct": 1.0, "result": result.result}

    if state == "FAILURE":
        return {"state": state, "stage": "FAILED", "message": str(result.info), "pct": 0.0}

    return {"state": state, "stage": state, "message": "", "pct": 0.0}
