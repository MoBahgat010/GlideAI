"""
FastAPI server for the multimodal RAG application.

Endpoints
─────────
POST  /api/ingest/chunk       — receive one file slice (aiofiles async write)
POST  /api/ingest/finalize    — reassemble slices, dispatch Celery ingestion job
GET   /api/ingest/status/{id} — poll Celery task progress
POST  /api/ask                — run retrieval + stream answer tokens (SSE)
GET   /api/health             — readiness probe
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path

import aiofiles
import aiofiles.os
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import (
    BASE_URL,
    CHUNKS_DIR,
    DEVICE,
    EMBED_BATCH,
    EMBEDDING_MODEL,
    HEAVY_WEIGHT,
    INDEX_NAME,
    LIGHT_WEIGHT,
    NVIDIA_API_KEY,
    QWEN_MODEL,
    QWEN_SERVER_URL,
    RERANK_TOP_K,
    RETRIEVE_TOP_K,
    UPLOAD_DIR,
    WEAVIATE_API_KEY,
    WEAVIATE_REST_ENDPOINT,
)

from ingestion.embedding import MultimodalEncoder
from retrieval.answer import AnswerGenerator
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import HybridReranker
from storage.weaviate import WeaviateVDB

from tasks import celery_app, run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server")

app = FastAPI(title="Multimodal RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retrieval_pipeline: RetrievalPipeline | None = None
_answer_generator: AnswerGenerator | None = None


def _get_retrieval() -> tuple[RetrievalPipeline, AnswerGenerator]:
    global _retrieval_pipeline, _answer_generator
    if _retrieval_pipeline is None or _answer_generator is None:
        logger.info("Initializing retrieval pipeline and answer generator...")
        reranker = HybridReranker()
        encoder = MultimodalEncoder(device=DEVICE, batch_size=EMBED_BATCH, model_name=EMBEDDING_MODEL)
        vdb = WeaviateVDB(
            endpoint=WEAVIATE_REST_ENDPOINT,
            api_key=WEAVIATE_API_KEY,
            index=INDEX_NAME,
            dimension=encoder.d_model,
        )

        qwen_base_url = QWEN_SERVER_URL.rstrip("/")
        if not qwen_base_url.endswith("/v1"):
            qwen_base_url += "/v1"

        logger.info("Connecting AsyncOpenAI client to QWEN_SERVER_URL: %s", qwen_base_url)
        client = AsyncOpenAI(api_key="EMPTY", base_url=qwen_base_url)
        model_name = QWEN_MODEL

        _retrieval_pipeline = RetrievalPipeline(
            encoder=encoder,
            vdb=vdb,
            local_client=client,
            local_model=model_name,
            retrieve_top_k=RETRIEVE_TOP_K,
            rerank_top_k=RERANK_TOP_K,
            reranker=reranker,
        )
        _answer_generator = AnswerGenerator(
            client=client,
            model=model_name,
        )
    return _retrieval_pipeline, _answer_generator



# ── health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── chunked upload ─────────────────────────────────────────────────────────────

@app.post("/api/ingest/chunk")
async def receive_chunk(
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),        # noqa: used for logging
    filename: str = Form(...),
):
    """
    Receive a single file slice and write it to disk with aiofiles.

    The frontend sends slices in order; the server stores them as
    ``<CHUNKS_DIR>/<upload_id>/<chunk_index:05d>.part``.
    """
    slot_dir = Path(CHUNKS_DIR) / upload_id
    await aiofiles.os.makedirs(str(slot_dir), exist_ok=True)

    chunk_path = slot_dir / f"{chunk_index:05d}.part"
    content = await file.read()

    async with aiofiles.open(str(chunk_path), "wb") as f:
        await f.write(content)

    logger.info(
        "Received chunk %d/%d for upload_id=%s  file=%s  bytes=%d",
        chunk_index + 1, total_chunks, upload_id, filename, len(content),
    )
    return {"upload_id": upload_id, "chunk_index": chunk_index, "received": True}


@app.post("/api/ingest/finalize")
async def finalize_upload(
    upload_id: str = Form(...),
    filename: str = Form(...),
    total_chunks: int = Form(...),
):
    """
    Reassemble all slices into the final file (async I/O), then dispatch
    a Celery ingestion task.
    """
    slot_dir = Path(CHUNKS_DIR) / upload_id
    upload_dir_path = Path(UPLOAD_DIR)
    upload_dir_path.mkdir(parents=True, exist_ok=True)
    final_path = upload_dir_path / filename

    logger.info(
        "Finalising upload_id=%s  filename=%s  total_chunks=%d",
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
    logger.info("Dispatched Celery task %s for job %s", task.id, job_id)

    return {"task_id": task.id, "job_id": job_id, "file": filename}


# ── task status polling ───────────────────────────────────────────────────────

@app.get("/api/ingest/status/{task_id}")
async def task_status(task_id: str):
    """Poll a Celery task for current state and progress metadata."""
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


# ── ask / streaming answer ────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str


@app.post("/api/ask")
async def ask(req: AskRequest):
    """
    Run retrieval then stream the LLM answer as Server-Sent Events.

    Each SSE message has the shape::

        data: {"type": "token", "content": "…"}\n\n

    A final ``{"type": "done", "results": […]}`` message closes the stream.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    pipeline, generator = _get_retrieval()

    async def event_stream():
        logger.info("Starting async retrieval for query=%r", query[:80])
        try:
            retrieval_result = await pipeline.retrieve(query)
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        results = retrieval_result.get("results", [])
        logger.info("Retrieval returned %d results; starting answer stream", len(results))

        # Stream LLM tokens asynchronously
        try:
            async for token in generator.stream(query, results):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Answer streaming failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        # Send results metadata as final event
        safe_results = _serialise_results(results)
        yield f"data: {json.dumps({'type': 'done', 'results': safe_results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _serialise_results(results: list[dict]) -> list[dict]:
    """Strip non-serialisable values from result dicts."""
    out = []
    for r in results:
        entry = {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, type(None)))}
        for nested_key in ("linked_image", "linked_caption"):
            if isinstance(r.get(nested_key), dict):
                entry[nested_key] = {
                    k: v for k, v in r[nested_key].items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
        out.append(entry)
    return out

