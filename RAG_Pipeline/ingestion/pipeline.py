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

    def run_pipeline(self, folder_path: str, user_id: str = "default"):
        docs = self.loader.load(folder_path)
        for doc in docs:
            chunks = self.chunker.chunk(doc, user_id=user_id)
            del doc
            gc.collect()
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                embed_results = self.encoder.embed_chunks(batch)
                if embed_results:
                    valid_chunks = [res["document"] for res in embed_results]
                    vectors = [res["embedding"] for res in embed_results]
                    self.vdb.upsert(valid_chunks, vectors)
                del batch, embed_results
                gc.collect()