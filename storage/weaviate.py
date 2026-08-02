"""
Weaviate vector store for the multimodal RAG pipeline.

Supports:
  - Automatic class creation (schema definition without auto-vectorization)
  - ``upsert_batch``: batched upsert using deterministic UUIDs from record IDs
  - ``hybrid_query`` / ``query``: hybrid vector + keyword search
  - ``fetch`` / ``fetch_batch``: retrieve records by ID (for linked content lookups)
"""

import json
import logging
import uuid
from typing import Any, Sequence

import weaviate

logger = logging.getLogger("storage.weaviate")


def _sanitize_class_name(name: str) -> str:
    """Convert index/class name to valid Weaviate PascalCase class name."""
    clean = "".join([c if c.isalnum() else " " for c in name]).strip()
    if not clean:
        return "RagPipeline"
    words = clean.split()
    class_name = "".join(w if w[0].isupper() else w.capitalize() for w in words)
    if not class_name[0].isupper():
        class_name = class_name[0].upper() + class_name[1:]
    return class_name


class WeaviateVDB:
    """
    Weaviate vector store wrapper for multimodal RAG.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str = "RagPipeline",
        dimension: int = 512,
    ):
        url = f"https://{endpoint}"
        
        self.class_name = _sanitize_class_name(index_name)
        self.dimension = dimension

        logger.info(
            "Connecting to Weaviate (url=%s, class_name=%s, dim=%d)…",
            url, self.class_name, self.dimension,
        )

        if hasattr(weaviate, "connect_to_weaviate_cloud"):
            try:
                from weaviate.auth import AuthApiKey
                auth = AuthApiKey(api_key) if api_key else None
            except (ImportError, AttributeError):
                auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None

            logger.info("Connecting via weaviate.connect_to_weaviate_cloud(cluster_url=%s)…", url)
            try:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=auth,
                )
            except Exception as exc:
                logger.warning("connect_to_weaviate_cloud failed (%s) — falling back to Client()", exc)
                auth_sec = weaviate.AuthApiKey(api_key=api_key) if api_key else None
                self.client = weaviate.Client(url=url, auth_client_secret=auth_sec)
        else:
            auth_sec = weaviate.AuthApiKey(api_key=api_key) if api_key else None
            self.client = weaviate.Client(url=url, auth_client_secret=auth_sec)

        is_ready = False
        if hasattr(self.client, "is_ready"):
            is_ready = self.client.is_ready()
        elif hasattr(self.client, "is_live"):
            is_ready = self.client.is_live()
        else:
            is_ready = True

        if not is_ready:
            raise ConnectionError(f"Weaviate cluster at {url} is not ready.")

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create class schema in Weaviate if it does not exist."""
        if self.client.schema.exists(self.class_name):
            logger.info("Using existing Weaviate class %r.", self.class_name)
            return

        # Check existing classes on cluster (e.g. 1-collection sandbox limit)
        existing = [c.get("class") for c in self.client.schema.get().get("classes", []) if c.get("class")]
        if existing:
            # Check case-insensitive match first
            matching = [c for c in existing if c.lower() == self.class_name.lower()]
            use_class = matching[0] if matching else existing[0]
            logger.warning(
                "Target class %r not found directly, using existing class %r (cluster has: %s).",
                self.class_name, use_class, existing,
            )
            self.class_name = use_class
            return

        logger.info("Creating Weaviate class %r…", self.class_name)
        schema = {
            "class": self.class_name,
            "vectorizer": "none",
            "vectorIndexConfig": {
                "distance": "cosine",
            },
            "properties": [
                {"name": "custom_id", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "chunk_text", "dataType": ["text"], "indexFilterable": True, "indexSearchable": True},
                {"name": "file_name", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "type", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "bbox", "dataType": ["number[]"]},
                {"name": "image_path", "dataType": ["string"]},
                {"name": "linked_image_id", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "linked_text_id", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
            ],
        }
        try:
            self.client.schema.create_class(schema)
            logger.info("Class %r created successfully.", self.class_name)
        except Exception as e:
            if "USAGE_LIMIT_EXCEEDED" in str(e) or "limit of 1 reached" in str(e):
                classes = [c.get("class") for c in self.client.schema.get().get("classes", []) if c.get("class")]
                if classes:
                    logger.warning("Class count limit reached. Falling back to existing class %r", classes[0])
                    self.class_name = classes[0]
                    return
            raise

    def recreate_class(self) -> None:
        """Delete existing class if present and recreate with current dimension."""
        try:
            existing = [c.get("class") for c in self.client.schema.get().get("classes", []) if c.get("class")]
            for cname in existing:
                logger.info("Deleting old class %r to reset schema for dim=%d…", cname, self.dimension)
                self.client.schema.delete_class(cname)
        except Exception as exc:
            logger.warning("Error clearing old class: %s", exc)

        self._ensure_schema()

    def _do_upsert(self, vectors: list[dict], batch_size: int = 128) -> None:
        with self.client.batch(batch_size=batch_size, dynamic=False) as batch:
            for item in vectors:
                custom_id = str(item["id"])
                vector = item.get("values")
                meta = item.get("metadata", {})

                if not vector or len(vector) != self.dimension:
                    logger.warning(
                        "Skipping vector id=%s: dimension mismatch or empty values (got %d, expected %d)",
                        custom_id, len(vector) if vector else 0, self.dimension
                    )
                    continue

                obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, custom_id))

                bbox_val = meta.get("bbox", "")
                if isinstance(bbox_val, (list, tuple, dict)):
                    bbox_val = json.dumps(bbox_val)
                elif bbox_val is None:
                    bbox_val = ""
                else:
                    bbox_val = str(bbox_val)

                properties = {
                    "custom_id": custom_id,
                    "chunk_text": str(meta.get("chunk_text") or ""),
                    "file_name": str(meta.get("file_name") or ""),
                    "type": str(meta.get("type") or ""),
                    "bbox": bbox_val,
                    "image_path": str(meta.get("image_path") or "") if meta.get("image_path") else None,
                    "linked_image_id": str(meta.get("linked_image_id") or "") if meta.get("linked_image_id") else None,
                    "linked_text_id": str(meta.get("linked_text_id") or "") if meta.get("linked_text_id") else None,
                }
                properties = {k: v for k, v in properties.items() if v is not None}

                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=obj_uuid,
                    vector=vector,
                )

    def upsert_batch(self, vectors: list[dict], batch_size: int = 128) -> None:
        """
        Upsert vectors in batches to Weaviate with automatic dimension mismatch recovery.

        Each vector: ``{"id": str, "values": list[float], "metadata": dict}``
        """
        if not vectors:
            return

        logger.info("Upserting %d vectors (dim=%d) to Weaviate class %s", len(vectors), self.dimension, self.class_name)

        try:
            self._do_upsert(vectors, batch_size)
        except Exception as exc:
            err_str = str(exc)
            if "vector with length" in err_str or "Existing nodes have vectors" in err_str:
                logger.warning(
                    "Dimension mismatch in Weaviate class %r: %s. Recreating class with dim=%d…",
                    self.class_name, exc, self.dimension,
                )
                self.recreate_class()
                self._do_upsert(vectors, batch_size)
            else:
                raise

        logger.info("Upsert completed for %d objects.", len(vectors))

    def hybrid_query(
        self,
        query_text: str = "",
        vector: list[float] | None = None,
        top_k: int = 30,
        alpha: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid vector + keyword search in Weaviate.

        Parameters
        ----------
        query_text: Keyword query text.
        vector: Dense query vector.
        top_k: Max results to return.
        alpha: Hybrid weight (0.0 = BM25 only, 1.0 = Vector search only).
        """
        props = [
            "custom_id", "chunk_text", "file_name", "type",
            "bbox", "image_path", "linked_image_id", "linked_text_id"
        ]

        query_builder = self.client.query.get(self.class_name, props)

        if vector and query_text:
            query_builder = query_builder.with_hybrid(
                query=query_text,
                vector=vector,
                alpha=alpha,
            )
        elif vector:
            query_builder = query_builder.with_near_vector({"vector": vector})
        elif query_text:
            query_builder = query_builder.with_bm25(query=query_text)
        else:
            return []

        query_builder = query_builder.with_additional(["id", "score", "distance", "certainty"])
        query_builder = query_builder.with_limit(top_k)

        resp = query_builder.do()

        if "errors" in resp:
            logger.error("Weaviate query error: %s", resp["errors"])
            return []

        get_data = resp.get("data", {}).get("Get", {}).get(self.class_name, [])

        results = []
        for item in get_data:
            add_info = item.get("_additional", {})
            raw_score = add_info.get("score") or add_info.get("certainty") or add_info.get("distance", 0.0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0

            res = {
                "id": item.get("custom_id") or add_info.get("id"),
                "score": score,
                "chunk_text": item.get("chunk_text") or "",
                "file_name": item.get("file_name") or "",
                "type": item.get("type") or "",
                "bbox": item.get("bbox") or "",
                "image_path": item.get("image_path") or None,
                "linked_image_id": item.get("linked_image_id") or None,
                "linked_text_id": item.get("linked_text_id") or None,
            }
            results.append(res)

        return results

    def query(
        self,
        vector: list[float],
        top_k: int = 30,
    ) -> list[dict]:
        """Backwards-compatible pure-vector search method."""
        return self.hybrid_query(query_text="", vector=vector, top_k=top_k, alpha=1.0)

    # ── fetch (for linked content lookups) ────────────────────────────────────

    def fetch(self, record_id: str) -> dict | None:
        """Fetch a single record by custom_id."""
        res = self.fetch_batch([record_id])
        return res[0] if res else None

    def fetch_batch(self, record_ids: list[str]) -> list[dict]:
        """Fetch multiple records by custom_id."""
        if not record_ids:
            return []

        props = [
            "custom_id", "chunk_text", "file_name", "type",
            "bbox", "image_path", "linked_image_id", "linked_text_id"
        ]

        operands = [{"path": ["custom_id"], "operator": "Equal", "valueString": rid} for rid in record_ids]
        where_clause = {
            "operator": "Or",
            "operands": operands,
        }

        resp = (
            self.client.query
            .get(self.class_name, props)
            .with_where(where_clause)
            .with_limit(len(record_ids))
            .do()
        )

        get_data = resp.get("data", {}).get("Get", {}).get(self.class_name, [])
        results = []
        for item in get_data:
            res = {
                "id": item.get("custom_id"),
                "chunk_text": item.get("chunk_text") or "",
                "file_name": item.get("file_name") or "",
                "type": item.get("type") or "",
                "bbox": item.get("bbox") or "",
                "image_path": item.get("image_path") or None,
                "linked_image_id": item.get("linked_image_id") or None,
                "linked_text_id": item.get("linked_text_id") or None,
            }
            results.append(res)
        return results
