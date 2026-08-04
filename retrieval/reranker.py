"""
Hybrid reranker: cross-encoder (BAAI/bge-reranker-base) + BM25.

Scores are independently min-max normalised then blended:
  final = alpha * ce_score + (1 - alpha) * bm25_score

Returns the top_k candidates sorted by combined score.
"""

import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger("retrieval.reranker")


class _CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        import torch
        self.model = CrossEncoder(model_name, model_kwargs={"torch_dtype": torch.float16})

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        import torch
        pairs = [[query, p] for p in passages]
        with torch.inference_mode():
            return np.array(self.model.predict(pairs, batch_size=32), dtype=float)



class _BM25Reranker:
    @staticmethod
    def score(query: str, passages: list[str]) -> np.ndarray:
        tokenized = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized)
        return np.array(bm25.get_scores(query.lower().split()), dtype=float)


class HybridReranker:
    """
    Combines BAAI/bge-reranker-base (cross-encoder) and BM25.

    Scores are independently min-max normalised then blended:
      final = alpha * ce_score + (1 - alpha) * bm25_score
    Returns the top_k candidates sorted by combined score.
    """

    def __init__(
        self,
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        alpha: float = 0.7,
    ):
        self.ce = _CrossEncoderReranker(cross_encoder_model)
        self.alpha = alpha

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = 5
    ) -> list[dict]:
        if not candidates:
            logger.info("No candidates to rerank.")
            return []

        passages = [c.get("chunk_text", "") for c in candidates]
        empty_count = sum(1 for p in passages if not p.strip())
        logger.info(
            "%d candidates, %d with empty chunk_text", len(candidates), empty_count
        )
        logger.debug("Sample passage (150 chars): %r", passages[0][:150])

        ce_scores = self.ce.score(query, passages)
        bm25_scores = _BM25Reranker.score(query, passages)

        logger.debug(
            "Cross-encoder — min=%.4f, max=%.4f, mean=%.4f",
            ce_scores.min(), ce_scores.max(), ce_scores.mean(),
        )
        logger.debug(
            "BM25           — min=%.4f, max=%.4f, mean=%.4f",
            bm25_scores.min(), bm25_scores.max(), bm25_scores.mean(),
        )

        combined = (
            self.alpha * _norm(ce_scores) + (1 - self.alpha) * _norm(bm25_scores)
        )
        logger.debug(
            "Combined       — min=%.4f, max=%.4f", combined.min(), combined.max()
        )

        top_idx = np.argsort(combined)[::-1][:top_k]
        results = [
            {**candidates[i], "rerank_score": float(combined[i])}
            for i in top_idx
        ]
        logger.info(
            "Top %d results: scores=%s",
            len(results),
            [round(r["rerank_score"], 4) for r in results],
        )
        return results



def _norm(arr: np.ndarray) -> np.ndarray:
    span = arr.max() - arr.min()
    return (arr - arr.min()) / span if span > 0 else arr
