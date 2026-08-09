"""
HybridReranker: Cross-encoder reranker powered by Triton Server (jinaai/jina-reranker-m0).
"""
import logging
from typing import Any

import numpy as np
import tritonclient.http as httpclient

logger = logging.getLogger("retrieval.reranker")


class HybridReranker:
    def __init__(self, url: str = "localhost:8000"):
        self.url = url
        self._client = None
        logger.info("Initializing HybridReranker connecting to Triton Server at %s", url)

    def _get_client(self):
        if self._client is None:
            self._client = httpclient.InferenceServerClient(url=self.url)
            logger.info("HybridReranker connected to Triton Server at %s", self.url)
        return self._client

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not candidates:
            logger.info("No candidates to rerank.")
            return []

        cand_ids = [c.get("custom_id") for c in candidates]
        logger.info("Reranking %d candidate chunks via Triton: %s", len(candidates), cand_ids)

        passages = [c.get("chunk_text") or c.get("caption") or "" for c in candidates]
        scores = self._score(query, passages)

        results = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            results.append(item)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = results[:top_k]
        logger.info(
            "Top %d results reranked via Triton: scores=%s",
            len(top_results),
            [round(r["rerank_score"], 4) for r in top_results],
        )
        return top_results

    def _score(self, query: str, passages: list[str]) -> list[float]:
        client = self._get_client()

        query_data = np.array([query.encode("utf-8")], dtype=object)
        passages_data = np.array([p.encode("utf-8") for p in passages], dtype=object)

        infer_query = httpclient.InferInput("QUERY", query_data.shape, "BYTES")
        infer_query.set_data_from_numpy(query_data)

        infer_passages = httpclient.InferInput("PASSAGES", passages_data.shape, "BYTES")
        infer_passages.set_data_from_numpy(passages_data)

        infer_out = httpclient.InferRequestedOutput("SCORES")

        response = client.infer(
            model_name="jina_reranker",
            inputs=[infer_query, infer_passages],
            outputs=[infer_out],
        )
        return response.as_numpy("SCORES").tolist()
