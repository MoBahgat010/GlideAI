import gc
import logging

import config
from config import EMBED_BATCH, INDEX_NAME, MAX_CHARS, OVERLAP_CHARS, TRITON_GRPC_URL
from RAG_Pipeline.ingestion.chunking import SemanticChunker
from RAG_Pipeline.ingestion.embedding import MultimodalEncoder
from RAG_Pipeline.ingestion.loader import PDFLoader
from RAG_Pipeline.storage import VDB, WeaviateVDB

logger = logging.getLogger("ingestion.pipeline")

class IngestionPipeline:
    def __init__(self, batch_size: int = EMBED_BATCH):
        self.loader = PDFLoader()
        self.chunker = SemanticChunker(MAX_CHARS, OVERLAP_CHARS)
        self.encoder = MultimodalEncoder(url=TRITON_GRPC_URL)

        self.vdb = VDB(strategy=WeaviateVDB(
                endpoint=config.WEAVIATE_REST_ENDPOINT,
                api_key=config.WEAVIATE_API_KEY,
                index=INDEX_NAME,
                dimension=self.encoder.d_model,
        ))
        # self.transcriber = RevAITranscriber()

        self.batch_size = batch_size
        logger.info("IngestionPipeline initialized with batch_size=%d", self.batch_size)

    def run_pipeline(self, folder_path: str, user_id: str = "default"):
        logger.info("Starting ingestion pipeline for path=%s user_id=%s", folder_path, user_id)
        docs = self.loader.load(folder_path)
        logger.info("Loaded %d document(s) from %s", len(docs), folder_path)

        for doc_idx, doc in enumerate(docs, 1):
            chunks = self.chunker.chunk(doc, user_id=user_id)
            total_chunks = len(chunks)
            logger.info(
                "Document %d/%d: Extracted %d total chunks for user_id=%s",
                doc_idx, len(docs), total_chunks, user_id,
            )

            del doc
            gc.collect()

            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size if total_chunks > 0 else 0
            logger.info("Processing %d total chunks in %d batch(es) (batch_size=%d)...", total_chunks, total_batches, self.batch_size)

            for b_idx, i in enumerate(range(0, total_chunks, self.batch_size), 1):
                batch = chunks[i : i + self.batch_size]
                logger.info(
                    "--> Batch %d/%d: Processing chunks %d..%d (%d chunks) via Triton & Weaviate...",
                    b_idx, total_batches, i, i + len(batch), len(batch),
                )
                vectors = self.encoder.embed_chunks(batch)
                self.vdb.upsert(batch, vectors)
                logger.info("Completed Batch %d/%d successfully.", b_idx, total_batches)
                del batch, vectors
                gc.collect()

        logger.info("Ingestion pipeline finished processing path=%s", folder_path)