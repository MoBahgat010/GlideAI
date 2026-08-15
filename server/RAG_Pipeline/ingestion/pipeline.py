import gc
import logging
import os
from pathlib import Path
from typing import Optional, List
from langchain_core.documents import Document
from config import EMBED_BATCH, REV_AI

from .chunking import SemanticChunker
from .embedding import MultimodalEncoder
from .loader import PDFLoader
from .stt import RevAITranscriber, is_media_file
from ..storage.vector_database import VDB

logger = logging.getLogger("ingestion.pipeline")


class IngestionPipeline:
    def __init__(
        self,
        loader: PDFLoader,
        chunker: SemanticChunker,
        encoder: MultimodalEncoder,
        vdb: VDB,
        batch_size: int = EMBED_BATCH,
    ):
        self.loader = loader
        self.chunker = chunker
        self.encoder = encoder
        self.vdb = vdb
        self.batch_size = batch_size

        self.transcriber = RevAITranscriber(access_token=REV_AI)
        logger.info("IngestionPipeline initialized with batch_size=%d (Rev AI active: %s)", self.batch_size, bool(self.transcriber))

    def run_pipeline(self, file_or_folder_path: str, session_id: str = "default"):
        """Process document/media file or entire session directory containing multiple uploaded files."""
        logger.info("Starting ingestion pipeline for path=%s, session_id=%s", file_or_folder_path, session_id)
        p = Path(file_or_folder_path)
        all_docs: List[Document] = []

        if p.is_dir():
            pdf_files = []
            media_files = []
            for child in p.iterdir():
                if child.is_file():
                    if is_media_file(child):
                        media_files.append(child)
                    elif child.suffix.lower() == ".pdf":
                        pdf_files.append(child)

            # Load PDFs
            if pdf_files:
                try:
                    logger.info("Loading %d PDF files from directory: %s", len(pdf_files), p)
                    pdf_docs = self.loader.load(str(p))
                    all_docs.extend(pdf_docs)
                except Exception as exc:
                    logger.warning("PDFLoader batch folder load failed (%s); trying individual files", exc)
                    for pdf_f in pdf_files:
                        try:
                            all_docs.extend(self.loader.load(str(pdf_f)))
                        except Exception as e:
                            logger.error("Failed to load PDF %s: %s", pdf_f, e)

            # Transcribe Media Files with Rev AI
            if media_files:
                if not self.transcriber:
                    logger.warning("Rev AI token not configured; skipping %d media file(s)", len(media_files))
                else:
                    for mf in media_files:
                        try:
                            logger.info("Transcribing media file via Rev AI: %s", mf.name)
                            transcript_docs = self.transcriber.load(str(mf))
                            all_docs.extend(transcript_docs)
                        except Exception as exc:
                            logger.error("Rev AI transcription failed for %s: %s", mf.name, exc)
        else:
            if is_media_file(p):
                if self.transcriber:
                    all_docs.extend(self.transcriber.load(str(p)))
                else:
                    logger.warning("Rev AI token not configured; cannot transcribe media file %s", p.name)
            else:
                all_docs.extend(self.loader.load(str(p)))

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