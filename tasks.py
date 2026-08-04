import logging
import sys
from pathlib import Path
from celery import Celery

# Ensure the project root is on sys.path for Celery's forked worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tasks")

from config import EMBED_BATCH, REDIS_URL
from ingestion.pipeline import IngestionPipeline

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)


ingestion_pipeline = IngestionPipeline(batch_size=EMBED_BATCH)

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

