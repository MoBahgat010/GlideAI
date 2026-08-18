import atexit
import gc
import logging
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone
import pymongo
from langchain_core.documents import Document

from src.db.mongo import mongo
from config import EMBED_BATCH, REV_AI
from .chunking import SemanticChunker
from .embedding import MultimodalEncoder
from .loader import PDFLoader
from .stt import RevAITranscriber, is_media_file
from ..storage.weaviate import WeaviateVDB
from ..storage.cloudinary_storage import cloudinary_client

logger = logging.getLogger("ingestion.pipeline")


class IngestionPipeline:
    def __init__(
        self,
        loader: PDFLoader,
        chunker: SemanticChunker,
        encoder: MultimodalEncoder,
        vdb: WeaviateVDB,
        batch_size: int = EMBED_BATCH,
    ):
        self.loader = loader
        self.chunker = chunker
        self.encoder = encoder
        self.vdb = vdb
        self.batch_size = batch_size

        self.db = mongo.sync_db
        self.transcriber = RevAITranscriber(access_token=REV_AI)
        logger.info("IngestionPipeline initialized with batch_size=%d", self.batch_size)

    def run_pipeline(self, file_or_folder_path: str, session_id: str):
        """Process document/media files, upload files to CDN, persist metadata in MongoDB, and index chunks in Weaviate."""
        logger.info("Starting ingestion pipeline for path=%s, session_id=%s", file_or_folder_path, session_id)
        p = Path(file_or_folder_path)
        all_docs: List[Document] = []
        files_to_process: List[Path] = []

        if p.is_dir():
            for child in p.iterdir():
                if child.is_file() and not child.name.endswith(".part"):
                    files_to_process.append(child)
        elif p.is_file():
            files_to_process.append(p)

        for src_file in files_to_process:
            file_name = src_file.name
            clean_stem = src_file.stem
            is_media = is_media_file(src_file)
            is_pdf = src_file.suffix.lower() == ".pdf"
            file_type = "pdf" if is_pdf else "media" if is_media else "document"

            public_id = f"{session_id}_{clean_stem}" if session_id else clean_stem
            file_url = cloudinary_client.upload_file(str(src_file), public_id=public_id)

            if session_id:
                file_meta = {
                    "filename": file_name,
                    "size": src_file.stat().st_size,
                    "file_type": file_type,
                    "file_url": file_url,
                    "url": file_url,
                    "uploaded_at": datetime.now(timezone.utc),
                }
                self.db.sessions.update_one(
                    {"session_id": session_id},
                    {"$pull": {"files": {"filename": file_name}}},
                )
                self.db.sessions.update_one(
                    {"session_id": session_id},
                    {"$push": {"files": file_meta}},
                )
                logger.info("Persisted file '%s' (%s) in MongoDB for session %s", file_name, file_url, session_id)

            if is_media:
                if self.transcriber:
                    loaded_docs = self.transcriber.load(str(src_file))
                else:
                    logger.warning("Rev AI not configured; skipping transcription for %s", file_name)
                    loaded_docs = []
            elif is_pdf:
                loaded_docs = self.loader.load(str(src_file))
            else:
                loaded_docs = []

            for doc in loaded_docs:
                doc.metadata["file_name"] = file_name
                doc.metadata["file_url"] = file_url
                all_docs.append(doc)

        logger.info("Total parsed document streams: %d for session_id=%s", len(all_docs), session_id)

        for doc_idx, doc in enumerate(all_docs, 1):
            chunks = self.chunker.chunk(doc, session_id=session_id)
            total_chunks = len(chunks)
            logger.info("Document stream %d/%d: Extracted %d chunks for session_id=%s", doc_idx, len(all_docs), total_chunks, session_id)

            del doc
            gc.collect()

            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size if total_chunks > 0 else 0

            for b_idx, i in enumerate(range(0, total_chunks, self.batch_size), 1):
                batch = chunks[i : i + self.batch_size]
                vectors = self.encoder.embed_chunks(batch)
                self.vdb.upsert(batch, vectors)
                logger.info("Indexed batch %d/%d (%d chunks) into Weaviate for session %s.", b_idx, total_batches, len(batch), session_id)
                del batch, vectors
                gc.collect()

        logger.info("Finished ingestion pipeline for path=%s", file_or_folder_path)