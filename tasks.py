"""
Celery tasks for the multimodal RAG pipeline.

run_ingestion:
  Downloads → parses/STT-transcribes → chunks → embeds → upserts to Pinecone.
  Reports stage progress via Celery's update_state for live Gradio streaming.
"""

import logging
import sys
from pathlib import Path

# Lazy imports so the worker only loads heavy models when a job starts
from ingestion.embedding import MultimodalEncoder, JinaClipTextEmbeddings
from ingestion.chunking import DocumentChunker
from ingestion.loader import PDFLoader
from ingestion.stt import RevAITranscriber
from ingestion.pipeline import IngestionPipeline
from storage.object_storage import ObjectStorage
from storage.pinecone import PineconeVDB

# Ensure the project root is on sys.path for Celery's forked worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tasks")

from celery import Celery

import rag_config as cfg

celery_app = Celery("rag", broker=cfg.REDIS_URL, backend=cfg.REDIS_URL)
celery_app.conf.task_track_started = True
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"


@celery_app.task(bind=True)
def run_ingestion(self, job_id: str, storage_keys: list[str]) -> dict:
    """
    Celery task that runs the full multimodal ingestion pipeline.

    Reports stage progress via update_state so the Gradio app can
    stream live status.
    """
    logger.info("═" * 60)
    logger.info("Ingestion job %s started — %d file(s)", job_id, len(storage_keys))
    logger.info("═" * 60)

    # ── Initialise components ─────────────────────────────────────────────────
    logger.info("Loading MultimodalEncoder (model=%s, device=%s)…",
                cfg.EMBEDDING_MODEL, cfg.EMBEDDING_DEVICE)
    encoder = MultimodalEncoder(device=cfg.EMBEDDING_DEVICE)

    logger.info("Connecting to Pinecone (index=%s, dim=%d)…",
                cfg.PINECONE_INDEX_NAME, encoder.get_dimension())
    vdb = PineconeVDB(
        cfg.PINECONE_API_KEY,
        cfg.PINECONE_INDEX_NAME,
        encoder.get_dimension(),
    )

    storage = ObjectStorage(s3_uri=cfg.S3_URI, local_dir=cfg.UPLOAD_DIR)

    lc_embeddings = JinaClipTextEmbeddings(encoder)
    chunker = DocumentChunker(
        embeddings=lc_embeddings,
        image_dir=cfg.IMAGE_DIR,
    )

    loader = PDFLoader(hybrid_url=cfg.HYBRID_URL, image_dir=cfg.IMAGE_DIR)
    transcriber = RevAITranscriber(
        access_token=cfg.REV_AI,
        poll_seconds=cfg.REV_AI_POLL_SECONDS,
        max_segment_seconds=cfg.STT_SEGMENT_SECONDS,
    )

    pipeline = IngestionPipeline(
        storage=storage,
        vdb=vdb,
        encoder=encoder,
        chunker=chunker,
        loader=loader,
        transcriber=transcriber,
    )

    # ── Progress callback → Celery state updates ─────────────────────────────
    def on_progress(stage: str, message: str, pct: float):
        logger.info("[%s] %.0f%% — %s", stage, pct * 100, message)
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "message": message, "pct": pct},
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        result = pipeline.run_pipeline(storage_keys, job_id, progress=on_progress)
        logger.info("═" * 60)
        logger.info("Ingestion job %s completed: %s", job_id, result)
        logger.info("═" * 60)
        return result
    except Exception as exc:
        logger.exception("Ingestion job %s FAILED", job_id)
        raise
