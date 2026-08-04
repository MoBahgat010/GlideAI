import gc
import logging

import config
from config import EMBED_BATCH, EMBEDDING_MODEL, INDEX_NAME, DEVICE, MAX_CHARS, OVERLAP_CHARS
from ingestion.chunking import SemanticChunker
from ingestion.embedding import MultimodalEncoder
from ingestion.loader import PDFLoader
from torch.cuda import empty_cache, ipc_collect
from ingestion.stt import RevAITranscriber, is_media_file
from storage.weaviate import WeaviateVDB

logger = logging.getLogger("ingestion.pipeline")

class IngestionPipeline:
    def __init__(self, batch_size: int = EMBED_BATCH):
        self.loader = PDFLoader()
        self.chunker = SemanticChunker(MAX_CHARS, OVERLAP_CHARS)
        self.encoder = MultimodalEncoder(DEVICE, EMBED_BATCH, EMBEDDING_MODEL)
        self.vdb = WeaviateVDB(
            endpoint=config.WEAVIATE_REST_ENDPOINT,
            api_key=config.WEAVIATE_API_KEY,
            index=INDEX_NAME,
            dimension=self.encoder.d_model,
        )
        # self.transcriber = RevAITranscriber()

        self.batch_size = batch_size

    def run_pipeline(self, folder_path: str):
        docs = self.loader.load(folder_path)
        for doc in docs:
            chunks = self.chunker.chunk(doc)
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
                empty_cache()
                ipc_collect()
                gc.collect()