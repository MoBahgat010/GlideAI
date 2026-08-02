"""
Streaming answer generation over retrieved RAG context using VLM (Async).

Constructs standard OpenAI multimodal message payload (text + image_url blocks)
passing retrieved real image URLs/paths to the VLM orchestrator model for
visual reasoning.
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import AsyncGenerator

from openai import AsyncOpenAI

logger = logging.getLogger("retrieval.answer")

_SYSTEM = """\
You are a precise, helpful multimodal assistant answering questions from retrieved document context and images.
Use ONLY the provided context and images. If the answer is not in the context or images, say so clearly.
Be concise and cite source files when relevant."""


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


def _build_vlm_content(query: str, results: list[dict]) -> tuple[list[dict], int]:
    """
    Format retrieved RAG context and images into standard VLM content parts.
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

    text_block = "\n\n".join(text_parts)
    user_prompt = f"Retrieved Context:\n{text_block}\n\nQuestion: {query}"

    content: list[dict] = [
        {"type": "text", "text": user_prompt}
    ]

    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    return content, len(image_urls)


class AnswerGenerator:
    """Streams answer tokens from the VLM API using AsyncOpenAI."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def stream(
        self,
        query: str,
        results: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Yield raw token strings asynchronously from the VLM.
        """
        user_content, num_images = _build_vlm_content(query, results)
        logger.info(
            "Generating VLM answer — query=%r  results=%d  images=%d  model=%s",
            query[:80], len(results), num_images, self.model,
        )

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ]

        response_stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
            stream=True,
        )

        total_tokens = 0
        async for chunk in response_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                total_tokens += 1
                yield delta.content

        logger.info("VLM stream complete — %d token chunks emitted", total_tokens)
