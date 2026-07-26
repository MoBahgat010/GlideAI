"""
Pinecone vector store for the multimodal RAG pipeline.

Creates the index automatically if it does not exist, using ``dimension``
derived at runtime from the actual embedding model output size.

Supports:
  - ``upsert_batch``: batched upsert (idempotent via deterministic IDs)
  - ``query``: vector similarity search with optional metadata filtering
  - ``fetch`` / ``fetch_batch``: retrieve records by ID (for linked content)
"""

import logging
import math
from collections.abc import Sequence
from time import sleep

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger("storage.pinecone")


class PineconeVDB:
    """
    Pinecone vector store.

    Creates the index automatically if it does not exist, using ``dimension``
    derived at runtime from the actual embedding model output size.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension

        existing = {i.name for i in self.pc.list_indexes()}
        if index_name not in existing:
            logger.info(
                "Creating Pinecone index %r (dim=%d, metric=%s)…",
                index_name, dimension, metric,
            )
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                sleep(1)
            logger.info("Index %r is ready.", index_name)
        else:
            logger.info("Using existing Pinecone index %r.", index_name)
            try:
                description = self.pc.describe_index(index_name)
                logger.info("Existing Pinecone index description: %s", description)
            except Exception as exc:
                logger.warning(
                    "Could not describe existing Pinecone index %r: %s",
                    index_name,
                    exc,
                )

        self.index = self.pc.Index(index_name)

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert_batch(self, vectors: list[dict], batch_size: int = 100) -> None:
        """
        Upsert vectors in batches.

        Each vector: ``{"id": str, "values": list[float], "metadata": dict}``

        Idempotent: upserting the same ID with the same values is a no-op.
        """
        logger.info(
            "Upserting %d vectors in batches of %d", len(vectors), batch_size
        )
        if vectors:
            sample_meta = vectors[0].get("metadata", {})
            logger.debug("Sample metadata keys: %s", list(sample_meta.keys()))
            chunk_text = sample_meta.get("chunk_text", "")
            logger.debug("Sample chunk_text (100 chars): %r", chunk_text[:100])

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            for vector in batch:
                values = self._validate_values(vector)
                vector["values"] = values
            logger.info(
                "Validated Pinecone batch %d: count=%d dims=%s ids=%s first_values=%s",
                i // batch_size + 1,
                len(batch),
                [len(v["values"]) for v in batch[:5]],
                [v.get("id") for v in batch[:5]],
                [v["values"][:3] for v in batch[:1]],
            )
            self.index.upsert(vectors=batch)
            logger.info(
                "Upserted batch %d/%d: %d vectors",
                i // batch_size + 1,
                (len(vectors) - 1) // batch_size + 1,
                len(batch),
            )

    # ── read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k: int = 30,
        filter: dict | None = None,
    ) -> list[dict]:
        """
        Query the index with a vector and return matches.

        Returns list of dicts with ``id``, ``score``, and all metadata fields.
        """
        kwargs: dict = {"vector": vector, "top_k": top_k, "include_metadata": True}
        if filter:
            kwargs["filter"] = filter
        resp = self.index.query(**kwargs)
        matches = resp.matches

        logger.info("Query returned %d matches", len(matches))
        if matches:
            first = matches[0]
            meta = first.metadata or {}
            logger.debug(
                "Top match — id=%r, score=%.4f, type=%s, meta_keys=%s",
                first.id, first.score,
                meta.get("type", "?"),
                list(meta.keys()),
            )

        results = []
        for m in matches:
            meta = m.metadata or {}
            if not meta.get("chunk_text"):
                logger.debug(
                    "Match %r has empty chunk_text (type=%s)",
                    m.id, meta.get("type", "?"),
                )
            results.append({"id": m.id, "score": m.score, **meta})
        return results

    # ── fetch (for linked content lookups) ────────────────────────────────────

    def fetch(self, record_id: str) -> dict | None:
        """
        Fetch a single record by ID.

        Returns a dict with ``id`` and all metadata fields, or None if not found.
        """
        resp = self.index.fetch(ids=[record_id])
        vectors = resp.get("vectors", {})
        if record_id not in vectors:
            logger.debug("fetch(%r) → not found", record_id)
            return None
        vec_data = vectors[record_id]
        meta = vec_data.get("metadata", {})
        logger.debug("fetch(%r) → type=%s", record_id, meta.get("type", "?"))
        return {"id": record_id, **meta}

    def fetch_batch(self, record_ids: list[str]) -> list[dict]:
        """
        Fetch multiple records by ID.

        Returns a list of dicts (only found records included).
        """
        if not record_ids:
            return []
        resp = self.index.fetch(ids=record_ids)
        vectors = resp.get("vectors", {})
        results = []
        for rid in record_ids:
            if rid in vectors:
                meta = vectors[rid].get("metadata", {})
                results.append({"id": rid, **meta})
        logger.debug(
            "fetch_batch: requested=%d, found=%d",
            len(record_ids), len(results),
        )
        return results

    def _validate_values(self, vector: dict) -> list[float]:
        values = vector.get("values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(
                "Refusing to upsert invalid vector values "
                f"for id={vector.get('id')!r}; expected a {self.dimension}-dim sequence."
            )

        values = list(values)
        logger.info(
            "Validating vector id=%r raw_type=%s raw_len=%d expected_dim=%d",
            vector.get("id"),
            type(vector.get("values")).__name__,
            len(values),
            self.dimension,
        )
        if len(values) != self.dimension:
            raise ValueError(
                "Refusing to upsert vector with wrong dimension "
                f"for id={vector.get('id')!r}: got {len(values)}, expected {self.dimension}."
            )

        try:
            values = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Refusing to upsert non-numeric vector values "
                f"for id={vector.get('id')!r}."
            ) from exc

        if not all(math.isfinite(v) for v in values):
            preview = values[:5]
            raise ValueError(
                "Refusing to upsert vector with NaN or infinite values "
                f"for id={vector.get('id')!r}; first_values={preview!r}."
            )

        return values
