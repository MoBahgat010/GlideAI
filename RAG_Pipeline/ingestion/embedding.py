"""
MultimodalEncoder: Bi-encoder for text and image embeddings powered by Triton Server (jinaai/jina-clip-v2).
"""
import base64
import binascii
import io
import logging
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from langchain_core.documents import Document
import tritonclient.http as httpclient

logger = logging.getLogger("ingestion.embedding")

_ENCODER_DIM = 1024


class MultimodalEncoder:
    d_model: int = _ENCODER_DIM

    def __init__(self, url: str):
        self.url = url
        self._client = None
        logger.info("Initializing MultimodalEncoder connecting to Triton Server at %s", url)

    def _get_client(self):
        if self._client is None:
            self._client = httpclient.InferenceServerClient(url=self.url)
            logger.info("MultimodalEncoder connected to Triton Server at %s", self.url)
        return self._client

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of strings via Triton → normalized float32 dense vectors."""
        client = self._get_client()

        safe_texts = [t if (t and t.strip()) else " " for t in texts]
        text_data = np.array([[t.encode("utf-8")] for t in safe_texts], dtype=object)

        infer_in = httpclient.InferInput("TEXT", text_data.shape, "BYTES")
        infer_in.set_data_from_numpy(text_data)

        infer_out = httpclient.InferRequestedOutput("EMBEDDING")

        response = client.infer(
            model_name="jina_encoder",
            inputs=[infer_in],
            outputs=[infer_out],
        )
        return response.as_numpy("EMBEDDING").tolist()

    def encode_image(self, images: list[Image.Image]) -> list[list[float]]:
        """Encode a batch of PIL Images via Triton → normalized float32 dense vectors."""
        client = self._get_client()

        b64_strings = [self._pil_to_base64(img) for img in images]
        image_data = np.array([[b.encode("utf-8")] for b in b64_strings], dtype=object)

        infer_in = httpclient.InferInput("IMAGE_BASE64", image_data.shape, "BYTES")
        infer_in.set_data_from_numpy(image_data)

        infer_out = httpclient.InferRequestedOutput("EMBEDDING")

        response = client.infer(
            model_name="jina_encoder",
            inputs=[infer_in],
            outputs=[infer_out],
        )
        return response.as_numpy("EMBEDDING").tolist()

    def embed_chunks(self, chunks: list[Document]) -> list[dict[str, Any]]:
        """
        Embed a mixed batch of text and image chunks via Triton.
        """
        results: list[dict[str, Any] | None] = [None] * len(chunks)

        text_indices: list[int] = []
        text_batch: list[str] = []

        image_indices: list[int] = []
        image_batch: list[Image.Image] = []

        for i, chunk in enumerate(chunks):
            if chunk.metadata.get("type") == "image":
                img_data = chunk.metadata.get("image_path") or chunk.metadata.get("image_base64")
                pil_image = self._decode_base64_image(img_data)
                if pil_image is None:
                    logger.warning("Skipping undecodable image chunk at index %d; using text fallback", i)
                    text_indices.append(i)
                    text_batch.append(chunk.page_content)
                else:
                    image_indices.append(i)
                    image_batch.append(pil_image)
            else:
                text_indices.append(i)
                text_batch.append(chunk.page_content)

        if text_batch:
            for idx, embedding in zip(text_indices, self.encode_text(text_batch)):
                results[idx] = {"document": chunks[idx], "embedding": embedding, "modality": "text"}

        if image_batch:
            for idx, embedding in zip(image_indices, self.encode_image(image_batch)):
                results[idx] = {"document": chunks[idx], "embedding": embedding, "modality": "image"}

        return [r for r in results if r is not None]

    @staticmethod
    def _pil_to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _decode_base64_image(data: str | None) -> Image.Image | None:
        if not data:
            return None
        if data.startswith("data:") and ";base64," in data:
            data = data.split(";base64,", 1)[1]
        try:
            raw = base64.b64decode(data, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as e:
            logger.warning("Failed to decode base64 image chunk: %s", e)
            return None