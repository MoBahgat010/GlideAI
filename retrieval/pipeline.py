import asyncio
import logging
from typing import Callable

from openai import AsyncOpenAI

from ingestion.embedding import MultimodalEncoder
from storage.weaviate import WeaviateVDB
from .query_rewriter import SingleCallQueryExpander, QueryRewriteAndHyDE
from .reranker import HybridReranker

logger = logging.getLogger("retrieval.pipeline")

Progress = Callable[[str, str], None]   # (stage, message)


class RetrievalPipeline:
    """
    Full multimodal retrieval pipeline using single-call Pydantic query expansion
    and asyncio.gather for parallel Weaviate hybrid retrieval.
    """

    def __init__(
        self,
        encoder: MultimodalEncoder,
        vdb: WeaviateVDB,
        local_client: AsyncOpenAI,
        local_model: str = "default",
        retrieve_top_k: int = 30,
        rerank_top_k: int = 5,
        reranker: HybridReranker | None = None,
    ):
        self.encoder = encoder
        self.vdb = vdb
        self.expander = SingleCallQueryExpander(local_client, local_model)
        self.reranker = reranker or HybridReranker()
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

    async def retrieve(self, query: str, progress: Progress | None = None) -> dict:
        """
        Run the full retrieval pipeline asynchronously.

        Returns
        -------
        dict with keys: ``query``, ``rewritten_query``, ``hyde_passage``, ``results``.
        """
        def emit(stage: str, msg: str):
            logger.info("[%s] %s", stage, msg)
            if progress:
                progress(stage, msg)

        logger.info("Retrieval started — query=%r", query)

        # ── 1. Single-call query rewrite + HyDE via local LLM & Pydantic ───────
        emit("EXPANDING", "Expanding query (rewrite + HyDE) in 1 call to local LLM…")
        expanded: QueryRewriteAndHyDE = await self.expander.expand(query)
        rewritten = expanded.rewritten_query
        hyde_passage = expanded.hyde_passage

        logger.info("Rewritten query: %r", rewritten[:120])
        logger.info("HyDE passage (120 ch): %r", hyde_passage[:120])

        # ── 2. Embed three query variants in one batch ─────────────────────────
        emit("EMBEDDING", "Embedding original / rewritten / HyDE queries…")
        orig_vec, rw_vec, hyde_vec = await asyncio.to_thread(
            self.encoder.encode_text,
            [query, rewritten, hyde_passage],
        )
        logger.debug("Query vectors encoded (dim=%d)", len(orig_vec))

        # ── 3. Hybrid search × 3 in parallel using asyncio.gather ─────────────
        emit("RETRIEVING", f"Hybrid querying Weaviate × 3 with asyncio.gather (top_k={self.retrieve_top_k})…")
        o_results, r_results, h_results = await asyncio.gather(
            asyncio.to_thread(self.vdb.hybrid_query, query_text=query,        vector=orig_vec,  top_k=self.retrieve_top_k, alpha=0.5),
            asyncio.to_thread(self.vdb.hybrid_query, query_text=rewritten,    vector=rw_vec,    top_k=self.retrieve_top_k, alpha=0.5),
            asyncio.to_thread(self.vdb.hybrid_query, query_text=hyde_passage, vector=hyde_vec,  top_k=self.retrieve_top_k, alpha=0.5),
        )

        seen: dict[str, dict] = {}
        for result_list in (o_results, r_results, h_results):
            for item in result_list:
                item_id = item.get("id")
                if not item_id:
                    continue
                if item_id not in seen or item.get("score", 0) > seen[item_id].get("score", 0):
                    seen[item_id] = item

        candidates = list(seen.values())
        logger.info("%d unique candidates after dedup", len(candidates))

        type_counts: dict[str, int] = {}
        for c in candidates:
            t = c.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("Candidate types: %s", type_counts)
        emit("RETRIEVING", f"{len(candidates)} candidates — types: {type_counts}")

        # ── 4. Rerank ──────────────────────────────────────────────────────────
        emit("RERANKING", f"Reranking → top {self.rerank_top_k}…")
        top_results = await asyncio.to_thread(
            self.reranker.rerank, query, candidates, top_k=self.rerank_top_k
        )
        logger.info("Rerank complete — %d results", len(top_results))
        for i, r in enumerate(top_results):
            logger.debug(
                "  [%d] type=%-10s score=%.4f id=%s",
                i, r.get("type", "?"), r.get("rerank_score", 0), r.get("id", "?"),
            )

        # ── 5. Enrich linked content ───────────────────────────────────────────
        emit("ENRICHING", "Fetching linked images/captions…")
        await asyncio.to_thread(self._enrich_linked_content, top_results, candidates)

        logger.info("Retrieval complete — returning %d results", len(top_results))
        return {
            "query": query,
            "rewritten_query": rewritten,
            "hyde_passage": hyde_passage,
            "results": top_results,
        }

    def _enrich_linked_content(
        self,
        results: list[dict],
        all_candidates: list[dict],
    ) -> None:
        """
        Cross-enrich caption↔image pairs in top results.
        """
        candidate_map: dict[str, dict] = {c["id"]: c for c in all_candidates if c.get("id")}
        caption_by_image: dict[str, dict] = {
            c["linked_image_id"]: c
            for c in all_candidates
            if c.get("type") == "caption" and c.get("linked_image_id")
        }

        ids_to_fetch: list[str] = []
        for r in results:
            if r.get("type") == "caption":
                lid = r.get("linked_image_id")
                if lid and lid not in candidate_map:
                    ids_to_fetch.append(lid)

        fetched: dict[str, dict] = {}
        if ids_to_fetch:
            logger.info("Fetching %d linked image records from Weaviate", len(ids_to_fetch))
            fetched = {
                rec["id"]: rec
                for rec in self.vdb.fetch_batch(ids_to_fetch)
                if rec.get("id")
            }

        for r in results:
            rtype = r.get("type")
            rid = r.get("id")

            if rtype == "caption":
                lid = r.get("linked_image_id")
                if lid:
                    linked = candidate_map.get(lid) or fetched.get(lid)
                    if linked:
                        r["linked_image"] = linked
                        logger.debug("Caption %s → linked image %s attached", rid, lid)
                    else:
                        logger.debug("Caption %s: linked image %s not found", rid, lid)

            elif rtype == "image":
                caption = caption_by_image.get(rid)
                if caption:
                    r["linked_caption"] = caption
                    logger.debug("Image %s → linked caption %s attached", rid, caption.get("id"))
