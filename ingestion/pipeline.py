"""
Ingestion pipeline — idempotent, multimodal, GPU-CPU pipelined.

Flow:
  1. Download all files in batch → temp dir
  2. Batch parse all PDF files together via OpenDataLoader / transcribe media
  3. Process each document with DocumentChunker → text + table + image + caption records
  4. Patch caption linked_image_id → full Weaviate custom_id
  5. Embed + upsert in overlapping GPU/CPU pipeline:
       GPU encodes batch[n+1]  while  CPU upserts batch[n] to Weaviate
"""

import gc
import json
import logging
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import config
from config import EMBED_BATCH, EMBEDDING_DEVICE, EMBEDDING_MODEL
from ingestion.chunking import DocumentChunker
from ingestion.embedding import MultimodalEncoder
from ingestion.loader import PDFLoader
from ingestion.models import ParsedDocument
from ingestion.stt import RevAITranscriber, is_media_file
from storage.object_storage import ObjectStorage
from storage.weaviate import WeaviateVDB

logger = logging.getLogger("ingestion.pipeline")

Progress = Callable[[str, str, float], None]  # (stage, message, pct)


def _build_vec_record(key: str, rec, raw_vec: list[float], expected_dim: int) -> dict | None:
    """Convert a chunk Document + raw embedding into a Weaviate upsert record."""
    try:
        vec = [float(v) for v in raw_vec]
    except (TypeError, ValueError):
        logger.warning("Non-numeric vector for %s::%s — skipping", key, rec.metadata.get("id"))
        return None

    if len(vec) != expected_dim:
        logger.warning(
            "Dimension mismatch for %s::%s: got %d expected %d — skipping",
            key, rec.metadata.get("id"), len(vec), expected_dim,
        )
        return None

    meta = {k: v for k, v in rec.metadata.items() if v is not None and k != "_vector"}
    if "bbox" in meta:
        meta["bbox"] = json.dumps(meta["bbox"])
    if rec.page_content:
        meta["chunk_text"] = rec.page_content

    return {
        "id": f"{key}::{rec.metadata['id']}",
        "values": vec,
        "metadata": meta,
    }


class IngestionPipeline:
    """
    End-to-end ingestion with batch PDF parsing and GPU-CPU pipeline overlap.
    """

    def __init__(
        self,
        storage: ObjectStorage,
        vdb: WeaviateVDB,
        encoder: MultimodalEncoder,
        chunker: DocumentChunker,
        loader: PDFLoader,
        transcriber: RevAITranscriber | None = None,
    ):
        self.storage = storage
        self.vdb = vdb
        self.encoder = encoder
        self.chunker = chunker
        self.loader = loader
        self.transcriber = transcriber

    @classmethod
    def from_config(cls) -> "IngestionPipeline":
        """Build pipeline from application config."""
        logger.info(
            "Loading MultimodalEncoder (model=%s  device=%s  batch_size=%d)",
            EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBED_BATCH,
        )
        encoder = MultimodalEncoder(device=EMBEDDING_DEVICE, batch_size=EMBED_BATCH)

        index_name = (
            getattr(config, "INDEX_NAME", None)
            or getattr(config, "PINECONE_INDEX_NAME", "RagPipeline")
        )
        logger.info(
            "Connecting to Weaviate (endpoint=%s  class=%s  dim=%d)",
            config.WEAVIATE_REST_ENDPOINT, index_name, encoder.get_dimension(),
        )
        vdb = WeaviateVDB(
            endpoint=config.WEAVIATE_REST_ENDPOINT,
            api_key=config.WEAVIATE_API_KEY,
            index_name=index_name,
            dimension=encoder.get_dimension(),
        )

        storage = ObjectStorage(s3_uri=config.S3_URI, local_dir=config.UPLOAD_DIR)
        chunker = DocumentChunker()
        loader = PDFLoader(image_dir=config.IMAGE_DIR)

        transcriber = None
        if getattr(config, "REV_AI", None):
            transcriber = RevAITranscriber(
                access_token=config.REV_AI,
                poll_seconds=getattr(config, "REV_AI_POLL_SECONDS", 10),
                max_segment_seconds=getattr(config, "STT_SEGMENT_SECONDS", 60),
            )

        return cls(
            storage=storage,
            vdb=vdb,
            encoder=encoder,
            chunker=chunker,
            loader=loader,
            transcriber=transcriber,
        )

    def _stream_text_embed_upsert(
        self,
        records: list,
        key: str,
        upsert_ex: ThreadPoolExecutor,
        pending: Future | None,
    ) -> tuple[int, Future | None]:
        expected_dim = self.encoder.get_dimension()
        batch_size = self.encoder.batch_size
        total_upserted = 0

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [r.page_content for r in batch]

            logger.info(
                "[TEXT] Encoding batch %d-%d/%d (key=%s)",
                start + 1, min(start + len(batch), len(records)), len(records), key,
            )
            raw_vecs = self.encoder.encode_text(texts)
            logger.debug("[TEXT] Encoded %d vectors (dim=%d)", len(raw_vecs), len(raw_vecs[0]) if raw_vecs else 0)

            weaviate_batch = []
            for rec, raw_vec in zip(batch, raw_vecs):
                rec_dict = _build_vec_record(key, rec, raw_vec, expected_dim)
                if rec_dict:
                    weaviate_batch.append(rec_dict)

            del texts, raw_vecs
            gc.collect()

            if weaviate_batch:
                if pending is not None:
                    pending.result()

                logger.info(
                    "[TEXT] GPU→CPU hand-off: submitting %d vectors to Weaviate (start=%d)",
                    len(weaviate_batch), start,
                )
                pending = upsert_ex.submit(self.vdb.upsert_batch, weaviate_batch)
                total_upserted += len(weaviate_batch)

        return total_upserted, pending

    def _stream_image_embed_upsert(
        self,
        records: list,
        key: str,
        upsert_ex: ThreadPoolExecutor,
        pending: Future | None,
    ) -> tuple[int, Future | None]:
        from PIL import Image as PILImage

        expected_dim = self.encoder.get_dimension()
        batch_size = max(1, self.encoder.batch_size // 2)
        total_upserted = 0

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]

            pil_images: list = []
            valid_recs: list = []
            for r in batch:
                path = r.metadata.get("image_path")
                if path and Path(path).exists():
                    try:
                        img = PILImage.open(path).convert("RGB")
                        pil_images.append(img)
                        valid_recs.append(r)
                    except Exception as exc:
                        logger.warning("Cannot open image %s: %s", path, exc)

            if not pil_images:
                continue

            logger.info(
                "[IMAGE] Encoding batch %d-%d/%d (key=%s)",
                start + 1, min(start + len(batch), len(records)), len(records), key,
            )
            raw_vecs = self.encoder.encode_image(pil_images)
            logger.debug("[IMAGE] Encoded %d vectors", len(raw_vecs))

            for img in pil_images:
                img.close()

            weaviate_batch = []
            for rec, raw_vec in zip(valid_recs, raw_vecs):
                rec_dict = _build_vec_record(key, rec, raw_vec, expected_dim)
                if rec_dict:
                    weaviate_batch.append(rec_dict)

            del pil_images, raw_vecs, valid_recs
            gc.collect()

            if weaviate_batch:
                if pending is not None:
                    pending.result()

                logger.info(
                    "[IMAGE] GPU→CPU hand-off: submitting %d image vectors to Weaviate (start=%d)",
                    len(weaviate_batch), start,
                )
                pending = upsert_ex.submit(self.vdb.upsert_batch, weaviate_batch)
                total_upserted += len(weaviate_batch)

        return total_upserted, pending

    def run_pipeline(
        self,
        storage_keys: list[str],
        job_id: str,
        progress: Progress | None = None,
    ) -> dict:
        """
        Run the ingestion pipeline. Batch parses all PDF files together in a
        single call to OpenDataLoader, then chunks, embeds, and upserts.
        """
        def emit(stage: str, msg: str, pct: float = 0.0):
            logger.info("[%s] %.0f%% — %s", stage, pct * 100, msg)
            if progress:
                progress(stage, msg, pct)

        total_files = len(storage_keys)
        stats = {
            "text_records": 0,
            "image_records": 0,
            "caption_records": 0,
            "total_records": 0,
            "job_id": job_id,
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # ── 1. Download all files ──────────────────────────────────────────
            local_files: list[tuple[str, Path, bool]] = []
            for file_idx, key in enumerate(storage_keys):
                emit("DOWNLOADING", f"[{file_idx+1}/{total_files}] Downloading {key}…", (file_idx / total_files) * 0.1)
                local = tmp_path / Path(key).name
                self.storage.download(key, local)
                is_media = is_media_file(local)
                local_files.append((key, local, is_media))
                logger.info("Downloaded %s → %s (%d bytes)", key, local, local.stat().st_size)

            # ── 2. Batch parse PDF files together ─────────────────────────────
            pdf_entries = [e for e in local_files if not e[2]]
            media_entries = [e for e in local_files if e[2]]

            doc_map: dict[str, ParsedDocument] = {}

            if pdf_entries:
                emit("PARSING", f"Batch parsing {len(pdf_entries)} PDF file(s) together…", 0.15)
                pdf_paths = [str(e[1]) for e in pdf_entries]
                try:
                    parsed_docs = self.loader.load(pdf_paths)
                    for (key, local, _), doc in zip(pdf_entries, parsed_docs):
                        doc_map[key] = doc
                except Exception as exc:
                    logger.error("Batch PDF parsing failed: %s — falling back to per-file parsing", exc)
                    for key, local, _ in pdf_entries:
                        try:
                            docs = self.loader.load([str(local)])
                            if docs:
                                doc_map[key] = docs[0]
                        except Exception as e2:
                            logger.error("Single PDF parse failed for %s: %s", key, e2)

            for key, local, _ in media_entries:
                emit("STT", f"Transcribing media {key}…", 0.20)
                try:
                    if self.transcriber is None:
                        raise RuntimeError("No STT transcriber configured.")
                    docs = self.transcriber.load([str(local)])
                    if docs:
                        doc_map[key] = docs[0]
                except Exception as exc:
                    logger.error("STT failed for %s: %s", key, exc)

            # ── 3. Chunk, embed, and upsert each document ─────────────────────
            for file_idx, (key, local, _) in enumerate(local_files):
                base_pct = 0.25 + (file_idx / total_files) * 0.75
                step = 0.75 / total_files

                doc = doc_map.get(key)
                if not doc or (not doc.page_content.strip() and not doc.kids):
                    logger.warning("No parsed content for %s — skipping", key)
                    local.unlink(missing_ok=True)
                    continue

                emit("CHUNKING", f"[{file_idx+1}/{total_files}] Chunking {key}", base_pct + step * 0.1)
                records = self.chunker.process_document(doc=doc)

                if not records:
                    logger.warning("No chunks produced for %s", key)
                    local.unlink(missing_ok=True)
                    continue

                image_elem_ids = {
                    r.metadata["id"]
                    for r in records
                    if r.metadata.get("type") == "image"
                }
                for r in records:
                    if r.metadata.get("type") == "caption":
                        raw_lid = r.metadata.get("linked_image_id")
                        if raw_lid is not None and raw_lid in image_elem_ids:
                            r.metadata["linked_image_id"] = f"{key}::{raw_lid}"

                text_recs = [r for r in records if r.metadata.get("type") != "image"]
                image_recs = [r for r in records if r.metadata.get("type") == "image"]

                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="weaviate-upsert") as upsert_ex:
                    pending: Future | None = None

                    if text_recs:
                        emit("EMBEDDING", f"[{file_idx+1}/{total_files}] Embedding {len(text_recs)} text records for {key}", base_pct + step * 0.4)
                        n, pending = self._stream_text_embed_upsert(text_recs, key, upsert_ex, pending)
                        stats["text_records"] += sum(1 for r in text_recs if r.metadata.get("type") not in ("caption",))
                        stats["caption_records"] += sum(1 for r in text_recs if r.metadata.get("type") == "caption")

                    if image_recs:
                        emit("EMBEDDING", f"[{file_idx+1}/{total_files}] Embedding {len(image_recs)} images for {key}", base_pct + step * 0.7)
                        n, pending = self._stream_image_embed_upsert(image_recs, key, upsert_ex, pending)
                        stats["image_records"] += len(image_recs)

                    if pending is not None:
                        pending.result()

                stats["total_records"] += len(records)
                del records, text_recs, image_recs
                gc.collect()
                local.unlink(missing_ok=True)

        emit("DONE", f"Pipeline complete — {stats['total_records']} records indexed.", 1.0)
        logger.info("✅ Job %s complete: %s", job_id, stats)
        return stats
