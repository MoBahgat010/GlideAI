"""
Multimodal Reranker using Svngoku/jina-reranker-m0-bnb-4bit.
Reranks candidates (text and images) directly using the cross-encoder model.
"""

import torch
import logging
from typing import Any
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("retrieval.reranker")


class HybridReranker:
    def __init__(self, model_name: str = "Svngoku/jina-reranker-m0-bnb-4bit"):
        self.model_name = model_name
        logger.info("Loading Jina Multimodal Reranker model: %s", model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not candidates:
            logger.info("No candidates to rerank.")
            return []

        cand_ids = [c.get("custom_id") for c in candidates]
        logger.info("Entering reranker with %d candidate chunk IDs: %s", len(candidates), cand_ids)

        passages = [c.get("chunk_text") or c.get("caption") or "" for c in candidates]
        pairs = [[query, p] for p in passages]

        with torch.inference_mode():
            if hasattr(self.model, "compute_score"):
                scores = self.model.compute_score(pairs)
            else:
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                scores = logits.squeeze(-1).cpu().numpy().tolist()

        if isinstance(scores, (float, int)):
            scores = [float(scores)]

        results = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            results.append(item)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = results[:top_k]
        logger.info(
            "Top %d results reranked with %s: scores=%s",
            len(top_results),
            self.model_name,
            [round(r["rerank_score"], 4) for r in top_results],
        )
        return top_results
