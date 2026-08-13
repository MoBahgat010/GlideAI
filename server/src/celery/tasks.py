import logging
import sys
from pathlib import Path
from celery import Celery
from .extraction_pompt import EXTRACTION_PROMPT

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tasks")

from config import (
    EMBED_BATCH,
    REDIS_URL,
    MONGODB_URL,
    MONGODB_DB_NAME,
    SUMMARIZER,
    BASE_URL,
    API_KEY,
    TRITON_HTTP_URL,
)
from RAG_Pipeline.ingestion.execute import ingestion_pipeline
import json
import redis
import pymongo
from openai import OpenAI
from urllib import request

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


@celery_app.task(bind=True)
def run_ingestion(self, job_id: str, storage_keys: str) -> dict:
    logger.info("Starting ingestion task for job_id=%s, storage_keys=%s", job_id, storage_keys)
    self.update_state(
        state="PROGRESS",
        meta={"stage": "PARSING", "message": "Loading document(s)...", "pct": 0.1},
    )

    try:
        self.update_state(
            state="PROGRESS",
            meta={"stage": "PROCESSING", "message": "Extracting text and chunking...", "pct": 0.3},
        )

        ingestion_pipeline.run_pipeline(storage_keys)

        self.update_state(
            state="PROGRESS",
            meta={"stage": "EMBEDDING", "message": "Indexing chunks to Weaviate...", "pct": 0.8},
        )

        result_meta = {"job_id": job_id, "status": "completed", "file": storage_keys}
        logger.info("Ingestion completed for job_id=%s", job_id)
        return result_meta

    except Exception as exc:
        logger.exception("Ingestion task failed for job_id=%s: %s", job_id, exc)
        raise exc


@celery_app.task(bind=True)
def extract_session_memory(self, session_id: str, user_id: str | None = None) -> dict:
    logger.info("Starting memory extraction for session_id=%s, user_id=%s", session_id, user_id)
    self.update_state(
        state="PROGRESS",
        meta={"stage": "FETCHING_HISTORY", "message": "Reading session chat history...", "pct": 0.2},
    )    

    key = f"session:{session_id}:working_memory"
    raw_history = redis_client.get(key)
    history = json.loads(raw_history) if raw_history else []

    if not history:
        logger.info("No history found in Redis for session_id=%s", session_id)
        return {"session_id": session_id, "status": "no_history"}

    formatted_transcript = "\n".join(
        [f"{item.get('role', 'user').upper()}: {item.get('content', '')}" for item in history]
    )

    self.update_state(
        state="PROGRESS",
        meta={"stage": "EXTRACTING_MEMORY", "message": "Running LLM memory extraction...", "pct": 0.5},
    )

    llmongo_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    extraction_prompt = EXTRACTION_PROMPT.format(formatted_transcript=formatted_transcript)

    try:
        response = llmongo_client.chat.completions.create(
            model=SUMMARIZER,
            messages=[{"role": "user", "content": extraction_prompt}],
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

        self.update_state(
            state="PROGRESS",
            meta={"stage": "PERSISTING", "message": "Saving memories to MongoDB...", "pct": 0.8},
        )

        doc_episodic = {
            "session_id": session_id,
            "user_id": user_id,
            "summary": parsed.get("episodic_summary", ""),
            "key_events": parsed.get("key_events", []),
            "created_at": pymongo.datetime.datetime.utcnow(),
        }
        db.episodic_memories.insert_one(doc_episodic)

        doc_semantic = {
            "session_id": session_id,
            "user_id": user_id,
            "facts": parsed.get("semantic_facts", []),
            "preferences": parsed.get("user_preferences", []),
            "created_at": pymongo.datetime.datetime.utcnow(),
        }
        db.semantic_memories.insert_one(doc_semantic)

        # Update session status
        db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "ended", "memory_extracted": True, "ended_at": pymongo.datetime.datetime.utcnow()}},
        )

        logger.info("Successfully extracted and persisted memories for session_id=%s", session_id)
        return {
            "session_id": session_id,
            "status": "success",
            "episodic_events_count": len(parsed.get("key_events", [])),
            "semantic_facts_count": len(parsed.get("semantic_facts", [])),
        }

    except Exception as exc:
        logger.exception("Memory extraction failed for session_id=%s: %s", session_id, exc)
        raise exc
