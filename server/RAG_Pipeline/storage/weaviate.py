import logging
from typing import Any, Optional
import weaviate
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.config import AdditionalConfig, Timeout
from weaviate.util import generate_uuid5

from config import EMBED_BATCH

from .vector_storage_strategy import VectorDatabaseStrategy

logger = logging.getLogger("storage.weaviate")

class WeaviateVDB(VectorDatabaseStrategy):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index: str,
        dimension: int,
    ):
        self.index = index
        self.dimension = dimension

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=endpoint,
            auth_credentials=Auth.api_key(api_key),
            additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
        )

        if not self.client.collections.exists(self.index):
            self.client.collections.create(
                name=self.index,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="custom_id", data_type=DataType.TEXT),
                    Property(name="session_id", data_type=DataType.TEXT),
                    Property(name="chunk_text", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="file_name", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="start_time", data_type=DataType.NUMBER),
                    Property(name="end_time", data_type=DataType.NUMBER),
                    Property(name="bbox", data_type=DataType.NUMBER_ARRAY),
                    Property(name="image_base64", data_type=DataType.TEXT),
                    Property(name="cloudinary_url", data_type=DataType.TEXT),
                    Property(name="linked_content_id", data_type=DataType.TEXT),
                ],
            )

        self.collection = self.client.collections.get(self.index)

    def as_vectorstore(
        self,
        embedding: Any = None,
        text_key: str = "chunk_text",
    ) -> WeaviateVectorStore:
        return WeaviateVectorStore(
            client=self.client,
            index_name=self.index,
            text_key=text_key,
            embedding=embedding,
        )

    def as_retriever(
        self,
        embedding: Any = None,
        text_key: str = "chunk_text",
        search_type: str = "similarity",
        **kwargs: Any,
    ):
        vectorstore = self.as_vectorstore(embedding=embedding, text_key=text_key)
        return vectorstore.as_retriever(search_type=search_type, **kwargs)

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if embeddings is not None:
            self.upsert(documents, embeddings)
            return [doc.metadata.get("custom_id") for doc in documents if doc.metadata.get("custom_id")]
        vectorstore = self.as_vectorstore(**kwargs)
        return vectorstore.add_documents(documents)

    def upsert(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        logger.info("Upserting %d chunks into Weaviate collection '%s'...", len(chunks), self.index)

        with self.collection.batch.fixed_size(batch_size=EMBED_BATCH, concurrent_requests=2) as batch:
            for chunk, embedding in zip(chunks, embeddings):
                if isinstance(embedding, dict) and "embedding" in embedding:
                    embedding = embedding["embedding"]

                logger.debug("Embedding dimensions: %s", len(embedding))

                metadata = dict(chunk.metadata)
                chunk_id = metadata.get("custom_id")

                properties = {
                    "custom_id": chunk_id,
                    "session_id": metadata.get("session_id"),
                    "chunk_text": chunk.page_content,
                    "type": metadata.get("type"),
                    "file_name": metadata.get("file_name"),
                    "page": metadata.get("page", None),
                    "start_time": metadata.get("start_time", None),
                    "end_time": metadata.get("end_time", None),
                    "bbox": metadata.get("bbox", None),
                    "image_base64": metadata.get("image_base64"),
                    "cloudinary_url": metadata.get("cloudinary_url"),
                    "linked_content_id": metadata.get("linked_content_id", None),
                }
                properties = {k: v for k, v in properties.items() if v is not None}

                obj_uuid = generate_uuid5(chunk_id)
                batch.add_object(
                    properties=properties,
                    vector=embedding,
                    uuid=obj_uuid,
                )

        if self.collection.batch.failed_objects:
            logger.error("Failed to upsert %d objects", len(self.collection.batch.failed_objects))

    def hybrid_query(
        self,
        query_text: str,
        vector: list[float],
        top_k: int,
        alpha: float,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid similarity/keyword search with session_id metadata filtering."""
        filters = None
        if session_id:
            filters = Filter.by_property("session_id").equal(session_id)

        response = self.collection.query.hybrid(
            query=query_text,
            vector=vector,
            alpha=alpha,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(score=True, distance=True),
        )

        results: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties)
            cid = props.get("custom_id") or str(obj.uuid)
            props["custom_id"] = cid
            props["weaviate_uuid"] = str(obj.uuid)
            if obj.metadata and obj.metadata.score is not None:
                props["score"] = obj.metadata.score
            results.append(props)

        return results

    def fetch_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        response = self.collection.query.fetch_objects(
            filters=Filter.by_property("custom_id").contains_any(ids),
            limit=len(ids),
        )

        results: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties)
            cid = props.get("custom_id") or str(obj.uuid)
            props["custom_id"] = cid
            props["weaviate_uuid"] = str(obj.uuid)
            results.append(props)

        return results