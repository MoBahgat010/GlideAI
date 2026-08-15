import logging
from typing import Any
import numpy as np
import tritonclient.grpc as grpcclient

from config import TRITON_GRPC_URL


logger = logging.getLogger("retrieval.reranker")


class HybridReranker:
    def __init__(self, url: str = TRITON_GRPC_URL):
        self._url = url
        self._client = grpcclient.InferenceServerClient(url=self._url)
        logger.info("HybridReranker connected to Triton gRPC at %s", url)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        cand_texts = [c.get("chunk_text") or c.get("caption") or "" for c in candidates]
        scores = self._score(query, cand_texts)

        results = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores)
        ]
        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return results[:top_k]

    def _score(self, query: str, candidates: list[str]) -> list[float]:
        query_data = np.array([[str(query)]], dtype=object)
        candidates_data = np.array([[str(c) for c in candidates]], dtype=object)

        infer_query = grpcclient.InferInput("QUERY", query_data.shape, "BYTES")
        infer_query.set_data_from_numpy(query_data)

        infer_candidates = grpcclient.InferInput("CANDIDATES", candidates_data.shape, "BYTES")
        infer_candidates.set_data_from_numpy(candidates_data)

        infer_out = grpcclient.InferRequestedOutput("SCORES")

        response = self._client.infer(
            model_name="cross_encoder",
            inputs=[infer_query, infer_candidates],
            outputs=[infer_out],
        )
        scores = response.as_numpy("SCORES")
        if scores is not None:
            return scores.flatten().tolist()
        return [0.0] * len(candidates)
