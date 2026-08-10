"""
Streaming Agentic RAG answer generation over retrieved RAG context using
LangChain's native `create_agent` from `langchain.agents` with Redis working memory per session.
"""

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import AsyncGenerator, Any

import redis.asyncio as aioredis
from openai import AsyncOpenAI

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

logger = logging.getLogger("retrieval.answer")

_AGENT_SYSTEM_PROMPT = """\
You are an intelligent, precise multimodal Agentic RAG assistant.
You process queries using retrieved document context, image data, and working conversation history.

Rules:
1. Rely primarily on the provided retrieved context and image data to answer questions.
2. Maintain context continuity using the conversation history provided.
3. Be clear, direct, concise, and cite original filenames or sections when appropriate.
4. If the retrieved context does not contain enough information, state that clearly while providing helpful guidance.
"""


def _encode_image_to_url(image_path: str) -> str | None:
    """Convert local image file path or web URL into a standard VLM image_url string."""
    if not image_path:
        return None
    if image_path.startswith("http://") or image_path.startswith("https://") or image_path.startswith("data:"):
        return image_path

    p = Path(image_path)
    if not p.exists():
        logger.warning("Image path does not exist on disk: %s", image_path)
        return None

    mime_type, _ = mimetypes.guess_type(str(p))
    if not mime_type:
        mime_type = "image/png"

    try:
        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"
    except Exception as e:
        logger.warning("Failed to encode image %s for VLM: %s", image_path, e)
        return None


def _build_vlm_content(query: str, results: list[dict], history: list[dict] | None = None) -> tuple[list[dict], int]:
    """
    Format retrieved RAG context, conversation history, and images into standard VLM content parts.
    Returns (content_list, num_images).
    """
    text_parts: list[str] = []
    image_urls: list[str] = []
    seen_images: set[str] = set()

    for i, r in enumerate(results, 1):
        rtype = r.get("type", "unknown")
        fname = r.get("file_name", "")
        text = r.get("chunk_text", "")

        # Text part of the block
        if rtype == "image":
            caption = ""
            linked = r.get("linked_caption", {}) or {}
            if linked:
                caption = linked.get("chunk_text", "")
            text_parts.append(f"[{i}] [IMAGE from {fname}] {caption}".strip())
        elif rtype == "caption":
            text_parts.append(f"[{i}] [CAPTION from {fname}] {text}")
        else:
            text_parts.append(f"[{i}] [{rtype.upper()} from {fname}] {text}")

        # Extract image URL / path for VLM
        img_path = r.get("image_path")
        if not img_path and r.get("linked_image"):
            img_path = r["linked_image"].get("image_path")

        if img_path and img_path not in seen_images:
            seen_images.add(img_path)
            url = _encode_image_to_url(img_path)
            if url:
                image_urls.append(url)

    context_block = "\n\n".join(text_parts)
    
    # Format working history
    history_block = ""
    if history:
        history_lines = []
        for msg in history[-6:]:  # Keep recent turns in prompt block
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_block = "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    user_prompt = f"{history_block}Retrieved Context:\n{context_block}\n\nCurrent Question: {query}"

    content: list[dict] = [
        {"type": "text", "text": user_prompt}
    ]

    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    return content, len(image_urls)


class RedisWorkingMemory:
    """Session-based working memory backed by Redis."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def get_history(self, session_id: str) -> list[dict]:
        """Fetch working chat history for a session."""
        try:
            client = await self._get_redis()
            key = f"session:{session_id}:working_memory"
            raw = await client.get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("Failed to fetch Redis working memory for %s: %s", session_id, e)
        return []

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message turn to Redis working memory."""
        try:
            client = await self._get_redis()
            key = f"session:{session_id}:working_memory"
            history = await self.get_history(session_id)
            history.append({"role": role, "content": content})
            # Keep up to last 30 turns in working memory
            history = history[-30:]
            await client.set(key, json.dumps(history), ex=86400 * 7)  # 7 day TTL
        except Exception as e:
            logger.warning("Failed to save Redis working memory for %s: %s", session_id, e)

    async def clear(self, session_id: str) -> None:
        """Clear session working memory."""
        try:
            client = await self._get_redis()
            key = f"session:{session_id}:working_memory"
            await client.delete(key)
        except Exception as e:
            logger.warning("Failed to clear Redis working memory for %s: %s", session_id, e)


class AgenticAnswerGenerator:
    """
    Agentic RAG answer generator built directly around `from langchain.agents import create_agent`
    and Redis session working memory.
    """

    def __init__(self, client: AsyncOpenAI, model: str, redis_url: str | None = None):
        self.client = client
        self.model = model
        self.working_memory = RedisWorkingMemory(redis_url) if redis_url else None
        
        base_url = str(client.base_url) if hasattr(client, "base_url") else None
        self.llm = ChatOpenAI(
            model=model,
            api_key=client.api_key or "EMPTY",
            base_url=base_url,
            temperature=0.2,
            streaming=True,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=_AGENT_SYSTEM_PROMPT,
        )
        logger.info("Initialized LangChain create_agent instance successfully.")

    async def stream(
        self,
        query: str,
        results: list[dict],
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent response while updating Redis working memory per session.
        """
        # Load working memory if session_id and Redis are available
        history: list[dict] = []
        if session_id and self.working_memory:
            history = await self.working_memory.get_history(session_id)

        user_content, num_images = _build_vlm_content(query, results, history=history)
        logger.info(
            "LangChain create_agent generating answer — session=%s query=%r results=%d images=%d model=%s history_turns=%d",
            session_id, query[:80], len(results), num_images, self.model, len(history),
        )

        messages = [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response_stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
            stream=True,
        )

        full_response_parts: list[str] = []
        total_tokens = 0

        async for chunk in response_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                total_tokens += 1
                full_response_parts.append(delta.content)
                yield delta.content

        full_response = "".join(full_response_parts)
        logger.info("LangChain create_agent stream complete — %d tokens emitted", total_tokens)

        # Save turns to Redis working memory
        if session_id and self.working_memory and full_response:
            await self.working_memory.add_message(session_id, "user", query)
            await self.working_memory.add_message(session_id, "assistant", full_response)
