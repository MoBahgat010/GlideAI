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
import uuid
from pathlib import Path

import aiofiles
import aiofiles.os
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import config
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

# ── directories ───────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(config.UPLOAD_DIR)
CHUNKS_DIR = UPLOAD_DIR / ".chunks"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── lazy retrieval pipeline (loaded once on first /ask) ───────────────────────

_retrieval_pipeline = None
_answer_generator = None


def _get_retrieval():
    global _retrieval_pipeline, _answer_generator
    if _retrieval_pipeline is not None:
        return _retrieval_pipeline, _answer_generator

    logger.info("Initialising retrieval pipeline (lazy)…")

    # Load reranker BEFORE encoder — see app.py comment about transformers registry
    from retrieval.reranker import HybridReranker
    reranker = HybridReranker()

    from ingestion.embedding import MultimodalEncoder
    from storage.weaviate import WeaviateVDB
    from retrieval.pipeline import RetrievalPipeline
    from retrieval.answer import AnswerGenerator
    from openai import AsyncOpenAI

    encoder = MultimodalEncoder(device=config.EMBEDDING_DEVICE, batch_size=config.EMBED_BATCH)

    index_name = (
        getattr(config, "INDEX_NAME", None)
        or getattr(config, "PINECONE_INDEX_NAME", "RagPipeline")
    )
    vdb = WeaviateVDB(
        endpoint=config.WEAVIATE_REST_ENDPOINT,
        api_key=config.WEAVIATE_API_KEY,
        index_name=index_name,
        dimension=encoder.get_dimension(),
    )

    local_base_url = config.LOCAL_LLM_URL
    if not local_base_url.endswith("/v1") and not local_base_url.endswith("/v1/"):
        local_base_url = f"{local_base_url.rstrip('/')}/v1"

    local_llm_client = AsyncOpenAI(api_key="local", base_url=local_base_url)
    nvidia_client = AsyncOpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL)

    _retrieval_pipeline = RetrievalPipeline(
        encoder=encoder,
        vdb=vdb,
        local_client=local_llm_client,
        retrieve_top_k=config.RETRIEVE_TOP_K,
        rerank_top_k=config.RERANK_TOP_K,
        reranker=reranker,
    )
    _answer_generator = AnswerGenerator(client=nvidia_client, model=config.HEAVY_WEIGHT_MODEL)

    logger.info("Retrieval pipeline ready (Local LLM: %s, VLM: %s).", local_base_url, config.HEAVY_WEIGHT_MODEL)
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
    slot_dir = CHUNKS_DIR / upload_id
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
    slot_dir = CHUNKS_DIR / upload_id
    final_path = UPLOAD_DIR / filename

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

    # Storage key is the filename relative to UPLOAD_DIR
    storage_key = filename
    job_id = upload_id

    task = run_ingestion.delay(job_id=job_id, storage_keys=[storage_key])
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
