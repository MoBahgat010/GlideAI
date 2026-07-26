"""
Retrieval pipeline for the multimodal RAG engine.

Full flow:
  1. Query rewrite (light LLM) + HyDE (heavy LLM) — in parallel
  2. Embed original / rewritten / HyDE queries via jina-clip-v2 text tower
  3. Query Pinecone with each vector (top_k=30) — in parallel, deduplicate
  4. Hybrid rerank (cross-encoder + BM25) → top 5
  5. Enrich results: follow linked_image_id / linked_text_id cross-references

Compatible with the ingestion pipeline: both use the same jina-clip-v2
text tower, producing vectors in the same 1024-dim shared space.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from openai import OpenAI

from ingestion.embedding import MultimodalEncoder
from storage.pinecone import PineconeVDB
from .query_rewriter import QueryRewriter, HyDEGenerator
from .reranker import HybridReranker

logger = logging.getLogger("retrieval.pipeline")

Progress = Callable[[str, str], None]   # (stage, message)


class RetrievalPipeline:
    """
    Full retrieval pipeline with multimodal support.

    At query time, a single text query vector naturally retrieves
    whichever hit — text chunk, image, or caption — ranks highest
    in the shared embedding space. Linked content is fetched for
    context/display.
    """

    def __init__(
        self,
        encoder: MultimodalEncoder,
        vdb: PineconeVDB,
        nvidia_client: OpenAI,
        light_model: str,
        heavy_model: str,
        retrieve_top_k: int = 30,
        rerank_top_k: int = 5,
        reranker: HybridReranker | None = None,
    ):
        self.encoder = encoder
        self.vdb = vdb
        self.nvidia_client = nvidia_client
        self.rewriter = QueryRewriter(nvidia_client, light_model)
        self.hyde = HyDEGenerator(nvidia_client, heavy_model)
        self.reranker = reranker or HybridReranker()
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: str, progress: Progress | None = None) -> dict:
        """
        Run the full retrieval pipeline.

        Returns
        -------
        dict
            Keys: query, rewritten_query, hyde_passage, results.
            Each result contains metadata with type, page_numbers, bboxes,
            image_path (if image), and linked content.
        """
        def emit(stage: str, msg: str):
            if progress:
                progress(stage, msg)

        logger.info("Retrieval started — query: %r", query)

        # ── 1. Rewrite + HyDE in parallel ─────────────────────────────────────
        emit("REWRITING", "Rewriting query and generating HyDE passage…")
        with ThreadPoolExecutor(max_workers=2) as ex:
            rw_fut = ex.submit(self.rewriter.rewrite, query)
            hy_fut = ex.submit(self.hyde.generate, query)
        rewritten = rw_fut.result()
        hyde_passage = hy_fut.result()
        logger.info("Rewritten: %r", rewritten[:120])
        logger.info("HyDE (120 chars): %r", hyde_passage[:120])
        emit("REWRITING", f"Rewritten: {rewritten[:120]}")

        # ── 2. Embed all three queries in one batch ────────────────────────────
        emit("EMBEDDING", "Embedding original / rewritten / HyDE queries…")
        orig_vec, rw_vec, hyde_vec = self.encoder.encode_text(
            [query, rewritten, hyde_passage]
        )

        # ── 3. Retrieve from Pinecone in parallel, deduplicate ─────────────────
        emit("RETRIEVING", f"Querying Pinecone (top_k={self.retrieve_top_k}) × 3…")
        with ThreadPoolExecutor(max_workers=3) as ex:
            o_res = ex.submit(self.vdb.query, orig_vec, self.retrieve_top_k)
            r_res = ex.submit(self.vdb.query, rw_vec,   self.retrieve_top_k)
            h_res = ex.submit(self.vdb.query, hyde_vec, self.retrieve_top_k)

        seen: dict[str, dict] = {}
        for result_list in [o_res.result(), r_res.result(), h_res.result()]:
            for item in result_list:
                if item["id"] not in seen or item["score"] > seen[item["id"]]["score"]:
                    seen[item["id"]] = item
        candidates = list(seen.values())
        logger.info("%d unique candidates after dedup", len(candidates))
        emit("RETRIEVING", f"{len(candidates)} unique candidates after dedup.")

        # Log record type distribution
        type_counts: dict[str, int] = {}
        for c in candidates:
            rtype = c.get("type", "unknown")
            type_counts[rtype] = type_counts.get(rtype, 0) + 1
        logger.info("Candidate types: %s", type_counts)

        # ── 4. Hybrid rerank → top 5 ───────────────────────────────────────────
        emit("RERANKING", f"Reranking with cross-encoder + BM25 → top {self.rerank_top_k}…")
        top_results = self.reranker.rerank(query, candidates, top_k=self.rerank_top_k)
        emit("RERANKING", f"Done. Returning {len(top_results)} results.")

        # ── 5. Enrich: follow linked content ───────────────────────────────────
        self._enrich_linked_content(top_results)

        logger.info(
            "Retrieval complete — %d results returned", len(top_results)
        )
        for i, r in enumerate(top_results):
            logger.debug(
                "  result[%d]: type=%s, score=%.4f, pages=%s",
                i, r.get("type", "?"), r.get("rerank_score", 0),
                r.get("page_numbers", []),
            )

        return {
            "query": query,
            "rewritten_query": rewritten,
            "hyde_passage": hyde_passage,
            "results": top_results,
        }

    def _enrich_linked_content(self, results: list[dict]) -> None:
        """
        For each result, fetch any linked content (image↔caption pairs)
        from Pinecone and attach it to the result dict.
        """
        linked_ids: list[str] = []
        for r in results:
            lid = r.get("linked_image_id") or r.get("linked_text_id")
            if lid:
                linked_ids.append(lid)

        if not linked_ids:
            return

        linked_records = self.vdb.fetch_batch(linked_ids)
        linked_map = {rec["id"]: rec for rec in linked_records}

        for r in results:
            if r.get("linked_image_id") and r["linked_image_id"] in linked_map:
                r["linked_image"] = linked_map[r["linked_image_id"]]
                logger.debug(
                    "Enriched result %s with linked image %s",
                    r["id"], r["linked_image_id"],
                )
            elif r.get("linked_text_id") and r["linked_text_id"] in linked_map:
                r["linked_text"] = linked_map[r["linked_text_id"]]
                logger.debug(
                    "Enriched result %s with linked text %s",
                    r["id"], r["linked_text_id"],
                )
