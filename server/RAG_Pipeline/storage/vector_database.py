from typing import Any
from langchain_core.documents import Document
from .vector_storage_strategy import VectorDatabaseStrategy

class VDB:
    def __init__(self, strategy: VectorDatabaseStrategy):
        self.strategy = strategy

    def as_vectorstore(self, embedding: Any = None, text_key: str = "chunk_text"):
        return self.strategy.as_vectorstore(embedding=embedding, text_key=text_key)

    def as_retriever(self, embedding: Any = None, text_key: str = "chunk_text", search_type: str = "similarity", **kwargs: Any):
        return self.strategy.as_retriever(embedding=embedding, text_key=text_key, search_type=search_type, **kwargs)

    def add_documents(self, documents: list[Document], embeddings: list[list[float]] | None = None, **kwargs: Any) -> list[str]:
        return self.strategy.add_documents(documents=documents, embeddings=embeddings, **kwargs)

    def upsert(self, chunks: list[Document], embeddings: list[list[float]]):
        return self.strategy.upsert(chunks=chunks, embeddings=embeddings)

    def hybrid_query(self, query_text: str, vector: list[float], top_k: int, alpha: float = 0.5, session_id: str = None) -> list[dict[str, Any]]:
        return self.strategy.hybrid_query(query_text=query_text, vector=vector, top_k=top_k, alpha=alpha, session_id=session_id)

    def fetch_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        return self.strategy.fetch_batch(ids=ids)