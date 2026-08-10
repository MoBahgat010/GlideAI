"""
HybridReranker: Cross-encoder reranker powered by Triton Server via gRPC.

Model: jinaai/jina-reranker-m0 (4-bit quantized, loaded inside Triton).

Input to Triton:
  QUERY      — shape [1],   single query string
  CANDIDATES — shape [-1],  array of candidate strings (text or image captions)

Output:
  SCORES — shape [-1], float32 relevance score per candidate.
"""
import logging
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient

from config import TRITON_GRPC_URL

logger = logging.getLogger("retrieval.reranker")


class HybridReranker:
    def __init__(self, url: str = TRITON_GRPC_URL):
        self._url = url
        self._client: grpcclient.InferenceServerClient | None = None
        logger.info("HybridReranker will connect to Triton gRPC at %s", url)

    # ─── gRPC client (lazy) ───────────────────────────────────────────────────

    def _get_client(self) -> grpcclient.InferenceServerClient:
        if self._client is None:
            self._client = grpcclient.InferenceServerClient(url=self._url)
            logger.info("HybridReranker connected to Triton gRPC at %s", self._url)
        return self._client

    # ─── Public API ───────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using the cross-encoder.

        Each candidate dict must have at least one of:
          - "chunk_text"  (text chunk)
          - "caption"     (image caption or alt-text)

        Returns the top_k candidates sorted by descending rerank_score.
        """
        if not candidates:
            logger.info("No candidates to rerank.")
            return []

        cand_ids = [c.get("custom_id") for c in candidates]
        logger.info("Reranking %d candidates via Triton: %s", len(candidates), cand_ids)

        # Extract text representations — never mutate the original dicts.
        cand_texts = [c.get("chunk_text") or c.get("caption") or "" for c in candidates]
        scores = self._score(query, cand_texts)

        results = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores)
        ]
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = results[:top_k]

        logger.info(
            "Top %d reranked via Triton: scores=%s",
            len(top_results),
            [round(r["rerank_score"], 4) for r in top_results],
        )
        return top_results

    # ─── Triton call ──────────────────────────────────────────────────────────

    def _score(self, query: str, candidates: list[str]) -> list[float]:
        query_data      = np.array([query.encode("utf-8")], dtype=object)
        candidates_data = np.array([c.encode("utf-8") for c in candidates], dtype=object)

        infer_query = grpcclient.InferInput("QUERY", query_data.shape, "TYPE_STRING")
        infer_query.set_data_from_numpy(query_data)

        infer_candidates = grpcclient.InferInput("CANDIDATES", candidates_data.shape, "TYPE_STRING")
        infer_candidates.set_data_from_numpy(candidates_data)

        infer_out = grpcclient.InferRequestedOutput("SCORES")

        response = self._get_client().infer(
            model_name="cross_encoder",
            inputs=[infer_query, infer_candidates],
            outputs=[infer_out],
        )
        return response.as_numpy("SCORES").tolist()
