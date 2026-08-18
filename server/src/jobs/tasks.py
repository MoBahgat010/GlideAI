import logging
from urllib import request
from celery import Celery

from config import REDIS_URL, TRITON_HTTP_URL
from server.RAG_Pipeline.ingestion.execute import ingestion_pipeline

logger = logging.getLogger("server.jobs.tasks")


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
def run_ingestion(self, job_id: str, storage_keys: str, session_id: str) -> dict:
    logger.info("Starting ingestion task job_id=%s, path=%s, session_id=%s", job_id, storage_keys, session_id)
    self.update_state(
        state="PROGRESS",
        meta={"stage": "PROCESSING", "message": "Processing and indexing documents...", "pct": 0.5},
    )

    try:
        ingestion_pipeline.run_pipeline(storage_keys, session_id=session_id)

        self.update_state(
            state="PROGRESS",
            meta={"stage": "EMBEDDING", "message": "Multi-modal indexing to Weaviate complete.", "pct": 1.0},
        )

        logger.info("Ingestion completed successfully for session_id=%s", session_id)
        return {
            "job_id": job_id,
            "status": "completed",
            "path": storage_keys,
            "session_id": session_id,
        }

    except Exception as exc:
        logger.exception("Ingestion task failed for session_id=%s: %s", session_id, exc)
        raise exc
