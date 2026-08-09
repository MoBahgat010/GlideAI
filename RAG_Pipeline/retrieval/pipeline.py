import asyncio
import logging
from typing import Callable, Any

from openai import AsyncOpenAI

from RAG_Pipeline.storage import VDB
from .query_rewriter import SingleCallQueryExpander

logger = logging.getLogger("retrieval.pipeline")

Progress = Callable[[str, str], None]   # (stage, message)


class RetrievalPipeline:
    def __init__(
        self,
        encoder: Any,
        vdb: VDB,
        local_client: AsyncOpenAI,
        local_model: str,
        retrieve_top_k: int,
        rerank_top_k: int,
        reranker: Any,
    ):
        self.encoder = encoder
        self.vdb = vdb
        self.expander = SingleCallQueryExpander(local_client, local_model)
        self.reranker = reranker
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
        # Uncomment below to re-enable HyDE & Query Rewriting:
        # emit("EXPANDING", "Expanding query (rewrite + HyDE) in 1 call to local LLM…")
        # expanded: QueryRewriteAndHyDE = await self.expander.expand(query)
        # rewritten = expanded.rewritten_query
        # hyde_passage = expanded.hyde_passage

        rewritten = query
        hyde_passage = ""

        # ── 2. Embed query variants ────────────────────────────────────────────
        emit("EMBEDDING", "Embedding query…")
        (orig_vec,) = await asyncio.to_thread(
            self.encoder.encode_text,
            [query],
        )
        logger.debug("Query vectors encoded (dim=%d)", len(orig_vec))

        # ── 3. Hybrid search in parallel using asyncio.gather ─────────────────
        emit("RETRIEVING", f"Hybrid querying Weaviate (top_k={self.retrieve_top_k})…")
        
        o_results = await asyncio.to_thread(
            self.vdb.hybrid_query, query_text=query, vector=orig_vec, top_k=self.retrieve_top_k, alpha=0.5
        )
        r_results = []
        h_results = []

        # orig_vec, rw_vec, hyde_vec = await asyncio.to_thread(self.encoder.encode_text, [query, rewritten, hyde_passage])
        # o_results, r_results, h_results = await asyncio.gather(
        #     asyncio.to_thread(self.vdb.hybrid_query, query_text=query,        vector=orig_vec,  top_k=self.retrieve_top_k, alpha=0.5),
        #     asyncio.to_thread(self.vdb.hybrid_query, query_text=rewritten,    vector=rw_vec,    top_k=self.retrieve_top_k, alpha=0.5),
        #     asyncio.to_thread(self.vdb.hybrid_query, query_text=hyde_passage, vector=hyde_vec,  top_k=self.retrieve_top_k, alpha=0.5),
        # )

        seen: dict[str, dict] = {}
        for result_list in (o_results, r_results, h_results):
            for item in result_list:
                item_id = item.get("custom_id")
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
        emit("RERANKING", f"Reranking {len(candidates)} candidates → top {self.rerank_top_k}…")
        top_results = await asyncio.to_thread(
            self.reranker.rerank, query, candidates, top_k=self.rerank_top_k
        )
        logger.info("Rerank complete — %d results", len(top_results))
        for i, r in enumerate(top_results):
            logger.debug(
                "  [%d] type=%-10s score=%.4f id=%s",
                i, r.get("type", "?"), r.get("rerank_score", 0), r.get("custom_id", "?"),
            )

        # ── 5. Enrich linked content ───────────────────────────────────────────
        emit("ENRICHING", "Fetching linked images/captions…")
        await self._enrich_linked_content(top_results, candidates)

        logger.info("Retrieval complete — returning %d results", len(top_results))
        return {
            "query": query,
            "rewritten_query": rewritten,
            "hyde_passage": hyde_passage,
            "results": top_results,
        }

    async def _enrich_linked_content(
        self,
        results: list[dict],
        all_candidates: list[dict],
    ) -> None:
        candidate_map: dict[str, dict] = {}
        for c in all_candidates:
            cid = c.get("custom_id")
            if cid:
                candidate_map[cid] = c

        ids_to_fetch: set[str] = set()
        for r in results:
            lid = r.get("linked_content_id")
            if lid and lid not in candidate_map:
                ids_to_fetch.add(lid)

        fetched_map: dict[str, dict] = {}
        if ids_to_fetch:
            logger.info("Fetching %d linked content records from Weaviate", len(ids_to_fetch))
            fetched_list = await asyncio.to_thread(self.vdb.fetch_batch, list(ids_to_fetch))
            for rec in fetched_list:
                rec_id = rec.get("custom_id")
                if rec_id:
                    fetched_map[rec_id] = rec

        for r in results:
            lid = r.get("linked_content_id")
            if lid:
                linked = candidate_map.get(lid) or fetched_map.get(lid)
                if linked:
                    linked_obj = dict(linked)
                    is_image = linked_obj.get("type") == "image"
                    if is_image:
                        b64 = linked_obj.get("image_base64")
                        r["linked_image"] = b64
                    r["linked_content"] = linked_obj
                    logger.debug("Attached linked content %s to chunk %s", lid, r.get("custom_id"))
                else:
                    logger.debug("Linked content %s not found for chunk %s", lid, r.get("custom_id"))