"""
Ingestion pipeline — idempotent, multimodal, batched.

Flow:
  1. Download files from object storage → temp dir
  2. Batch-load PDFs with OpenDataLoaderPDFLoader (single call)
  3. Process each document with DocumentChunker → text + image + caption records
  4. Embed text/caption records with jina-clip-v2 text tower
  5. Embed image records with jina-clip-v2 image tower
  6. Upsert all records to Pinecone (idempotent via deterministic IDs)
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Callable

import rag_config as cfg

from ingestion.loader import PDFLoader
from ingestion.chunking import DocumentChunker
from ingestion.embedding import MultimodalEncoder
from storage.object_storage import ObjectStorage
from storage.pinecone import PineconeVDB

logger = logging.getLogger("ingestion.pipeline")

Progress = Callable[[str, str, float], None]  # (stage, message, pct 0-1)


class IngestionPipeline:
    """
    End-to-end ingestion: download → parse → chunk → embed → index.

    Idempotent: deterministic chunk IDs mean re-running with the same
    input produces the same records → Pinecone upsert is a no-op.
    """

    def __init__(
        self,
        storage: ObjectStorage,
        vdb: PineconeVDB,
        encoder: MultimodalEncoder,
        chunker: DocumentChunker,
        loader: PDFLoader,
    ):
        self.storage = storage
        self.vdb = vdb
        self.encoder = encoder
        self.chunker = chunker
        self.loader = loader

    def run_pipeline(
        self,
        storage_keys: list[str],
        job_id: str,
        progress: Progress | None = None,
    ) -> dict:
        """
        Run the full ingestion pipeline file-by-file to minimize memory usage.
        """
        import gc

        def emit(stage: str, msg: str, pct: float = 0.0):
            if progress:
                progress(stage, msg, pct)

        total_files = len(storage_keys)
        summary_stats = {
            "text_records": 0,
            "image_records": 0,
            "caption_records": 0,
            "total_records": 0,
            "job_id": job_id,
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            for file_idx, key in enumerate(storage_keys):
                base_pct = file_idx / total_files
                pct_step = 1.0 / total_files

                emit("DOWNLOADING", f"File {file_idx+1}/{total_files}: Downloading {key}…", base_pct + pct_step * 0.05)
                local = tmp_path / Path(key).name
                self.storage.download(key, local)
                logger.info("Downloaded %s → %s (%d bytes)", key, local, local.stat().st_size)

                # ── Parse entire file directly (Temporary single-step) ──────────
                emit("PARSING", f"File {file_idx+1}/{total_files}: Parsing document…", base_pct + pct_step * 0.10)
                
                try:
                    documents = self.loader.load([str(local)])
                except Exception as e:
                    logger.error("Docling failed: %s", e)
                    documents = None
                    
                if not documents:
                    continue
                
                doc = documents[0]
                parsed = doc.page_content
                if not parsed or not parsed.strip():
                    logger.warning(
                        "Skipping empty parse result for %s",
                        key
                    )
                    del doc
                    del documents
                    gc.collect()
                    continue
                
                emit("CHUNKING", f"File {file_idx+1}/{total_files}: Chunking document…", base_pct + pct_step * 0.30)
                records = self.chunker.process_document(
                    parsed_json=parsed,
                    source_key=key,
                    job_id=job_id,
                    page_offset=0,
                )
                
                del parsed
                del doc
                del documents
                
                if not records:
                    gc.collect()
                    continue

                # ── Embed Text + Captions for this chunk ──────────────────────
                text_embeddable = [r for r in records if r["record_type"] in ("text", "caption")]
                if text_embeddable:
                    emit("EMBEDDING", f"File {file_idx+1}/{total_files}: Embedding {len(text_embeddable)} text records…", base_pct + pct_step * 0.50)
                    texts = [r["text"] for r in text_embeddable]
                    logger.info(
                        "Text records selected for embedding: ids=%s char_counts=%s",
                        [r.get("id") for r in text_embeddable[:5]],
                        [len(r.get("text") or "") for r in text_embeddable[:5]],
                    )
                    text_vectors = self.encoder.encode_text(texts)
                    logger.info(
                        "Text embedding returned %d vectors; lengths=%s",
                        len(text_vectors),
                        [len(v) if hasattr(v, "__len__") else "unknown" for v in text_vectors[:5]],
                    )
                    if len(text_vectors) != len(text_embeddable):
                        raise ValueError(
                            "Text embedding count mismatch: "
                            f"got {len(text_vectors)} vectors for {len(text_embeddable)} records."
                        )
                    for rec, vec in zip(text_embeddable, text_vectors):
                        rec["_vector"] = vec
                    del texts
                    del text_vectors
                    gc.collect()

                # ── Embed Images for this chunk ───────────────────────────────
                image_recs = [r for r in records if r["record_type"] == "image"]
                if image_recs:
                    emit("EMBEDDING", f"File {file_idx+1}/{total_files}: Embedding {len(image_recs)} images…", base_pct + pct_step * 0.70)
                    from PIL import Image
                    
                    img_batch_size = 4
                    for i in range(0, len(image_recs), img_batch_size):
                        batch_recs = image_recs[i : i + img_batch_size]
                        images = []
                        valid_recs = []
                        
                        for r in batch_recs:
                            if r.get("image_path") and Path(r["image_path"]).exists():
                                try:
                                    img = Image.open(r["image_path"]).convert("RGB")
                                    images.append(img)
                                    valid_recs.append(r)
                                except Exception as e:
                                    logger.warning("Failed to open image %s: %s", r["image_path"], e)
                        
                        if images:
                            image_vectors = self.encoder.encode_image(images, batch_size=2)
                            logger.info(
                                "Image embedding returned %d vectors; lengths=%s",
                                len(image_vectors),
                                [len(v) if hasattr(v, "__len__") else "unknown" for v in image_vectors[:5]],
                            )
                            if len(image_vectors) != len(valid_recs):
                                raise ValueError(
                                    "Image embedding count mismatch: "
                                    f"got {len(image_vectors)} vectors for {len(valid_recs)} records."
                                )
                            for rec, vec in zip(valid_recs, image_vectors):
                                rec["_vector"] = vec
                            
                            for img in images:
                                img.close()
                            
                            del images
                            del image_vectors
                            gc.collect()

                # ── Upsert to Pinecone for this chunk ─────────────────────────
                emit("UPSERTING", f"File {file_idx+1}/{total_files}: Uploading {len(records)} vectors…", base_pct + pct_step * 0.85)
                pinecone_vecs = []
                expected_dim = self.encoder.get_dimension()
                for rec in records:
                    raw_vec = rec.get("_vector")
                    if raw_vec is None:
                        logger.warning(
                            "Skipping record %s (%s) with no embedding vector",
                            rec.get("id"),
                            rec.get("record_type"),
                        )
                        continue

                    try:
                        vec = [float(v) for v in raw_vec]
                    except (TypeError, ValueError):
                        logger.warning(
                            "Skipping record %s (%s) with non-numeric vector",
                            rec.get("id"),
                            rec.get("record_type"),
                        )
                        continue

                    if len(vec) != expected_dim:
                        logger.warning(
                            "Skipping record %s (%s) with invalid vector length %s (expected %s)",
                            rec.get("id"),
                            rec.get("record_type"),
                            len(vec),
                            expected_dim,
                        )
                        continue

                    meta = {k: v for k, v in rec["metadata"].items() if v is not None}
                    for key_m in ("page_numbers", "bboxes"):
                        if key_m in meta and isinstance(meta[key_m], list):
                            if key_m == "bboxes":
                                meta[key_m] = json.dumps(meta[key_m])
                            elif key_m == "page_numbers":
                                # Pinecone lists must contain strings, not numbers
                                meta[key_m] = [str(int(p)) if isinstance(p, float) else str(p) for p in meta[key_m]]

                    pinecone_vecs.append({
                        "id": rec["id"],
                        "values": vec,
                        "metadata": meta,
                    })

                logger.info(
                    "Prepared Pinecone vectors: count=%d lengths=%s ids=%s",
                    len(pinecone_vecs),
                    [len(v["values"]) for v in pinecone_vecs[:5]],
                    [v["id"] for v in pinecone_vecs[:5]],
                )
                if pinecone_vecs:
                    self.vdb.upsert_batch(pinecone_vecs)
                else:
                    logger.warning(
                        "No valid Pinecone vectors prepared for %s; records=%d",
                        key,
                        len(records),
                    )

                # Update stats
                summary_stats["text_records"] += sum(1 for r in records if r["record_type"] == "text")
                summary_stats["image_records"] += len(image_recs)
                summary_stats["caption_records"] += sum(1 for r in records if r["record_type"] == "caption")
                summary_stats["total_records"] += len(records)

                # Clear this chunk's data from memory completely
                del records
                del text_embeddable
                del image_recs
                del pinecone_vecs
                gc.collect()
                local.unlink(missing_ok=True)


            emit("DONE", f"Done. {summary_stats['total_records']} total records indexed.", 1.0)
            logger.info("✅ Pipeline complete for job %s: %s", job_id, summary_stats)
            return summary_stats
