"""
Query rewriting and HyDE generation for the retrieval pipeline.

Uses NVIDIA NIM API (OpenAI-compatible client):
  - QueryRewriter  — light model, keyword-enriched rewrite
  - HyDEGenerator  — heavy model, hypothetical answer passage
"""

import logging
import time

from openai import OpenAI

logger = logging.getLogger("retrieval.rewriter")


class QueryRewriter:
    """Rewrites a user query to be more retrieval-friendly (LIGHT model)."""

    _SYSTEM = (
        "Rewrite the following search query to improve document retrieval. "
        "Make it specific and keyword-rich. Return only the rewritten query — no explanation."
    )

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def rewrite(self, query: str) -> str:
        logger.info("Rewriting query with model=%s …", self.model)
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.3,
            timeout=30,
        )
        elapsed = time.time() - t0
        content = None
        if getattr(resp, "choices", None):
            content = resp.choices[0].message.content
            
        logger.info("Rewrite completed in %.1fs — %r", elapsed, content)
        if content is None:
            logger.warning("Rewriter returned None — using original query")
            return query
        return content.strip()


class HyDEGenerator:
    """
    Hypothetical Document Embedding — generates a passage that *would* answer
    the query, then embeds it for retrieval (HEAVY model for richer output).
    """

    _SYSTEM = (
        "Write a concise, factual passage (3–5 sentences) that would directly answer "
        "the question below, as if extracted from a real document. "
        "Return only the passage — no preamble."
    )

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate(self, query: str) -> str:
        logger.info("Generating HyDE passage with model=%s …", self.model)
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=400,
            temperature=0.5,
            timeout=60,
        )
        elapsed = time.time() - t0
        content = None
        if getattr(resp, "choices", None):
            content = resp.choices[0].message.content
            
        logger.info("HyDE completed in %.1fs — %r", elapsed, content[:120] if content else None)
        if content is None:
            logger.warning("HyDE returned None — using original query")
            return query
        return content.strip()
