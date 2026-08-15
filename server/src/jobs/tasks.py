import logging
import json
from pathlib import Path
import sys
from urllib import request
from celery import Celery
import redis
import pymongo
from openai import OpenAI

from server.RAG_Pipeline.ingestion.execute import ingestion_pipeline
from .extraction_pompt import EXTRACTION_PROMPT
from config import (
    REDIS_URL,
    MONGODB_URL,
    MONGODB_DB_NAME,
    SUMMARIZER,
    BASE_URL,
    API_KEY,
    TRITON_HTTP_URL,
    ENABLE_SEMANTIC_MEMORY,
    ENABLE_EPISODIC_MEMORY,
)

logger = logging.getLogger("server.jobs.tasks")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
mongo_client = pymongo.MongoClient(MONGODB_URL)
db = mongo_client[MONGODB_DB_NAME]

def check_triton_health(url: str) -> bool:
    health_url = f"{url}/v2/health/ready"
    try:
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False

if not check_triton_health(TRITON_HTTP_URL):
    raise ConnectionError(f"Triton server is not running at {TRITON_HTTP_URL}")

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)


@celery_app.task(name="run_ingestion", bind=True)
def run_ingestion(self, job_id: str, storage_keys: str, session_id: str = "default") -> dict:
    """Asynchronously process document & media files for a session and upsert to Weaviate with session filtering."""
    logger.info("Starting ingestion task job_id=%s, path=%s, session_id=%s", job_id, storage_keys, session_id)
    self.update_state(
        state="PROGRESS",
        meta={"stage": "PARSING", "message": "Parsing document layout & transcribing media via Rev AI...", "pct": 0.2},
    )

    try:
        self.update_state(
            state="PROGRESS",
            meta={"stage": "PROCESSING", "message": "Semantic chunking & Cloudinary visual uploads...", "pct": 0.5},
        )

        ingestion_pipeline.run_pipeline(storage_keys, session_id=session_id)

        self.update_state(
            state="PROGRESS",
            meta={"stage": "EMBEDDING", "message": "Multi-modal indexing to Weaviate complete.", "pct": 1.0},
        )

        logger.info("Ingestion completed successfully for session_id=%s", session_id)
        return {"job_id": job_id, "status": "completed", "path": storage_keys, "session_id": session_id}

    except Exception as exc:
        logger.exception("Ingestion task failed for session_id=%s: %s", session_id, exc)
        raise exc


@celery_app.task(name="extract_session_memory", bind=True)
def extract_session_memory(self, session_id: str, user_id: str = "default") -> dict:
    """Extract episodic events and semantic facts from session transcript based on feature flags."""
    logger.info("Starting memory extraction for session_id=%s (flags: semantic=%s, episodic=%s)", session_id, ENABLE_SEMANTIC_MEMORY, ENABLE_EPISODIC_MEMORY)

    if not ENABLE_SEMANTIC_MEMORY and not ENABLE_EPISODIC_MEMORY:
        logger.info("Memory extraction feature flags are disabled. Skipping extraction for session %s", session_id)
        db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "ended", "memory_extracted": False, "ended_at": pymongo.datetime.datetime.utcnow()}},
        )
        return {"session_id": session_id, "status": "skipped_disabled"}

    self.update_state(
        state="PROGRESS",
        meta={"stage": "FETCHING_HISTORY", "message": "Reading conversation transcript...", "pct": 0.2},
    )

    key = f"session:{session_id}:working_memory"
    raw_history = redis_client.get(key)
    history = json.loads(raw_history) if raw_history else []

    if not history:
        logger.info("No history found in Redis for session_id=%s", session_id)
        db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "ended", "memory_extracted": False, "ended_at": pymongo.datetime.datetime.utcnow()}},
        )
        return {"session_id": session_id, "status": "no_history"}

    formatted_transcript = "\n".join(
        [f"{item.get('role', 'user').upper()}: {item.get('content', '')}" for item in history]
    )

    self.update_state(
        state="PROGRESS",
        meta={"stage": "EXTRACTING_MEMORY", "message": "Synthesizing episodic & semantic memories...", "pct": 0.6},
    )

    llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = EXTRACTION_PROMPT.format(formatted_transcript=formatted_transcript)

    try:
        response = llm_client.chat.completions.create(
            model=SUMMARIZER,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""

        parsed = {}
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(content[start:end])
        except Exception:
            parsed = {
                "episodic_summary": content,
                "key_events": [],
                "semantic_facts": [],
                "user_preferences": [],
            }

        now = pymongo.datetime.datetime.utcnow()

        if ENABLE_EPISODIC_MEMORY:
            doc_episodic = {
                "session_id": session_id,
                "user_id": user_id,
                "summary": parsed.get("episodic_summary", ""),
                "key_events": parsed.get("key_events", []),
                "created_at": now,
            }
            db.episodic_memories.replace_one({"session_id": session_id}, doc_episodic, upsert=True)

        if ENABLE_SEMANTIC_MEMORY:
            doc_semantic = {
                "session_id": session_id,
                "user_id": user_id,
                "facts": parsed.get("semantic_facts", []),
                "preferences": parsed.get("user_preferences", []),
                "created_at": now,
            }
            db.semantic_memories.replace_one({"session_id": session_id}, doc_semantic, upsert=True)

        db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "ended", "memory_extracted": True, "ended_at": now}},
        )

        logger.info("Successfully persisted extracted memories for session_id=%s", session_id)
        return {
            "session_id": session_id,
            "status": "success",
            "episodic_events": len(parsed.get("key_events", [])),
            "semantic_facts": len(parsed.get("semantic_facts", [])),
        }

    except Exception as exc:
        logger.exception("Memory extraction failed for session_id=%s: %s", session_id, exc)
        raise exc
