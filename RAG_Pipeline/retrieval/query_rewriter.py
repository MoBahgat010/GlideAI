"""
Query rewriting and HyDE generation for the retrieval pipeline in a SINGLE LLM call (Async).

Uses local LLM (e.g., Qwen server on http://127.0.0.1:8080) with a Pydantic model
to generate both the rewritten query and the HyDE passage in ONE single API call.
"""

import json
import logging
import time

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("retrieval.rewriter")


class QueryRewriteAndHyDE(BaseModel):
    """Pydantic model representing both rewritten query and HyDE passage."""
    rewritten_query: str = Field(
        description="Search-optimized, keyword-rich rewritten version of the user query."
    )
    hyde_passage: str = Field(
        description="A concise, factual hypothetical document passage (3-5 sentences) that directly answers the query."
    )


class SingleCallQueryExpander:
    """
    Expands a user query into both a search-optimized rewrite AND a HyDE passage
    in a SINGLE call to the local LLM using a Pydantic model.
    """

    _SYSTEM = (
        "You are an expert search query expansion system. Given a user query, perform two tasks:\n"
        "1. Produce a search-optimized, keyword-rich rewritten version of the query.\n"
        "2. Produce a hypothetical document passage (3–5 sentences) that directly answers the query.\n\n"
        "Respond ONLY with valid JSON matching the schema."
    )

    def __init__(self, client: AsyncOpenAI, model: str = "default"):
        self.client = client
        self.model = model

    async def expand(self, query: str) -> QueryRewriteAndHyDE:
        """
        Generate both rewritten_query and hyde_passage in ONE single LLM call.
        """
        logger.info("Expanding query (rewrite + HyDE) in 1 single call to local LLM (%s)…", self.client.base_url)
        t0 = time.time()

        prompt = (
            f"User Query: {query}\n\n"
            "Return JSON in this format:\n"
            "{\n"
            '  "rewritten_query": "search optimized version of query",\n'
            '  "hyde_passage": "hypothetical factual answer passage"\n'
            "}"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                top_p=0.3,
                timeout=30,
            )
            elapsed = time.time() - t0
            raw_content = resp.choices[0].message.content or "{}"
            data = json.loads(raw_content)

            result = QueryRewriteAndHyDE(
                rewritten_query=data.get("rewritten_query") or query,
                hyde_passage=data.get("hyde_passage") or query,
            )
            logger.info(
                "Query expansion finished in %.2fs — rewrite: %r, hyde: %r",
                elapsed, result.rewritten_query[:60], result.hyde_passage[:60],
            )
            return result

        except Exception as exc:
            elapsed = time.time() - t0
            logger.warning(
                "Local LLM expansion failed after %.2fs (%s) — falling back to original query",
                elapsed, exc,
            )
            return QueryRewriteAndHyDE(rewritten_query=query, hyde_passage=query)


class QueryRewriter:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.expander = SingleCallQueryExpander(client, model)

    async def rewrite(self, query: str) -> str:
        res = await self.expander.expand(query)
        return res.rewritten_query


class HyDEGenerator:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.expander = SingleCallQueryExpander(client, model)

    async def generate(self, query: str) -> str:
        res = await self.expander.expand(query)
        return res.hyde_passage
