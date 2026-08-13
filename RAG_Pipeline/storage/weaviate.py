import logging
import uuid
from typing import Any

import weaviate
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5
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
        )

        if not self.client.collections.exists(self.index):
            self.client.collections.create(
                name=self.index,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="custom_id", data_type=DataType.TEXT),
                    Property(name="chunk_text", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="file_name", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="bbox", data_type=DataType.NUMBER_ARRAY),
                    Property(name="image_base64", data_type=DataType.TEXT),
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
            return [
                doc.metadata.get("custom_id")
                for doc in documents
            ]

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

        with self.collection.batch.dynamic() as batch:
            for chunk, embedding in zip(chunks, embeddings):
                if isinstance(embedding, dict) and "embedding" in embedding:
                    embedding = embedding["embedding"]

                metadata = dict(chunk.metadata)
                chunk_id = metadata.get("custom_id")

                properties = {
                    "custom_id": chunk_id,
                    "chunk_text": chunk.page_content,
                    "type": metadata.get("type"),
                    "file_name": metadata.get("file_name"),
                    "page": metadata.get("page"),
                    "bbox": metadata.get("bbox"),
                    "image_base64": metadata.get("image_base64") or metadata.get("image_path"),
                    "linked_content_id": metadata.get("linked_content_id"),
                }

                properties = {k: v for k, v in properties.items() if v is not None}

                obj_uuid = generate_uuid5(chunk_id)

                batch.add_object(
                    properties=properties,
                    vector=embedding,
                    uuid=obj_uuid,
                )

        logger.info("Finished upserting batch of %d chunks to Weaviate (deterministic UUIDs enforced).", len(chunks))

    def hybrid_query(
        self,
        query_text: str,
        vector: list[float],
        top_k: int,
        alpha: float,
    ) -> list[dict[str, Any]]:
        response = self.collection.query.hybrid(
            query=query_text,
            vector=vector,
            alpha=alpha,
            limit=top_k,
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