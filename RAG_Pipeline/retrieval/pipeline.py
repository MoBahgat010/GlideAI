import asyncio
import logging
from typing import Callable, Any

from openai import AsyncOpenAI

from RAG_Pipeline.storage import VDB
from RAG_Pipeline.ingestion.embedding import MultimodalEncoder
from .reranker import HybridReranker

logger = logging.getLogger("retrieval.pipeline")

Progress = Callable[[str, str], None]   # (stage, message)


class RetrievalPipeline:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        reranker: HybridReranker,
        vdb: VDB,
        retrieve_top_k: int,
        rerank_top_k: int,
    ):
        self.encoder = encoder
        self.vdb = vdb
        self.reranker = reranker
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

    async def retrieve(self, query: str) -> dict:
        embedding = self.encoder.encode_query(query)
        candidates = self.vdb.hybrid_query(query, embedding, self.retrieve_top_k, 0.5)
        top_results = self.reranker.rerank(query, candidates, self.rerank_top_k)

        await self._enrich_linked_content(top_results, candidates)

        return {
            "query": query,
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