from abc import ABC, abstractmethod
from typing import Any
from langchain_core.documents import Document


class VectorDatabaseStrategy(ABC):
    @abstractmethod
    def as_vectorstore(self, embedding: Any = None, text_key: str = "chunk_text"):
        pass

    @abstractmethod
    def as_retriever(self, embedding: Any = None, text_key: str = "chunk_text", search_type: str = "similarity", **kwargs: Any):
        pass

    @abstractmethod
    def add_documents(self, documents: list[Document], embeddings: list[list[float]] | None = None, **kwargs: Any) -> list[str]:
        pass

    @abstractmethod
    def upsert(self, chunks: list[Document], embeddings: list[list[float]]):
        pass

    @abstractmethod
    def hybrid_query(self, query_text: str, vector: list[float], top_k: int, alpha: float) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def fetch_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        pass
