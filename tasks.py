"""
Celery tasks for the multimodal RAG pipeline.

run_ingestion
  Downloads → parses/STT-transcribes → chunks → embeds (GPU) → upserts (CPU,
  overlapped with GPU) → Weaviate.  Reports stage progress via Celery
  update_state so the FastAPI /status endpoint can surface live progress.

IMPORTANT: always start the worker with --pool=solo to avoid forking the
CUDA context and loading the model N times:

    celery -A tasks worker --pool=solo --loglevel=info
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path for Celery's forked worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tasks")

from celery import Celery  # noqa: E402

import config                                      # noqa: E402
from ingestion.pipeline import IngestionPipeline   # noqa: E402

celery_app = Celery("rag", broker=config.REDIS_URL, backend=config.REDIS_URL)
celery_app.conf.task_track_started = True
celery_app.conf.task_serializer   = "json"
celery_app.conf.result_serializer = "json"

# ── Pipeline singleton — loaded once per worker process, reused across jobs ───
# With --pool=solo this is the same process for every task, so the model is
# loaded exactly once (no forking, no CUDA corruption, minimal RAM usage).
_pipeline: IngestionPipeline | None = None


def _get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Loading IngestionPipeline for the first time in this worker…")
        _pipeline = IngestionPipeline.from_config()
        logger.info("Pipeline ready — model will be reused for subsequent jobs.")
    return _pipeline


@celery_app.task(bind=True)
def run_ingestion(self, job_id: str, storage_keys: list[str]) -> dict:
    """
    Run the full multimodal ingestion pipeline inside a Celery worker.

    GPU embedding and Weaviate upsert are overlapped: each finished embedding
    batch is handed to a background thread for upsert while the GPU processes
    the next batch.  Garbage-collects aggressively throughout.

    Start the worker with --pool=solo so the model is loaded once:
        celery -A tasks worker --pool=solo --loglevel=info
    """
    logger.info("═" * 60)
    logger.info("Ingestion job %s — %d file(s): %s", job_id, len(storage_keys), storage_keys)
    logger.info("═" * 60)

    def on_progress(stage: str, message: str, pct: float) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "message": message, "pct": pct},
        )

    try:
        pipeline = _get_pipeline()
        result = pipeline.run_pipeline(storage_keys, job_id, progress=on_progress)
        logger.info("═" * 60)
        logger.info("Job %s done: %s", job_id, result)
        logger.info("═" * 60)
        return result
    except Exception:
        logger.exception("Job %s FAILED", job_id)
        raise
