import base64
import binascii
import io
import logging
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("ingestion.embedding")

class MultimodalEncoder:
    def __init__(self, device: str, batch_size: int, model_name: str):
        self.batch_size = batch_size

        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True, "device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )

        self.image_embeddings = self.text_embeddings._client
        self.d_model = self.image_embeddings.get_sentence_embedding_dimension()

        logger.info("Loaded %s (dim=%s) on device=%s", model_name, self.d_model, self.image_embeddings.device)


    def encode_text(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [t if (t and t.strip()) else " " for t in texts]
        with torch.inference_mode():
            return self.text_embeddings.embed_documents(safe_texts)

    def encode_image(self, images: list[Image.Image]) -> list[list[float]]:
        with torch.inference_mode():
            embeddings = self.image_embeddings.encode(
                images,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.tolist()


    def embed_chunks(self, chunks: list[Document]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(chunks)

        text_indices: list[int] = []
        text_batch: list[str] = []

        image_indices: list[int] = []
        image_batch: list[Image.Image] = []

        for i, chunk in enumerate(chunks):
            # if chunk.metadata.get("type") == "image":
            #     pil_image = self._decode_base64_image(chunk.metadata.get("image_path"))
            #     if pil_image is None:
            #         logger.warning("Skipping image chunk at index %d: could not decode base64 image data", i)
            #         continue
            #     image_indices.append(i)
            #     image_batch.append(pil_image)
            # else:
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
    def _decode_base64_image(data: str | None) -> Image.Image | None:
        if not data:
            return None

        if data.startswith("data:") and ";base64," in data:
            data = data.split(";base64,", 1)[1]

        try:
            raw = base64.b64decode(data, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as e:
            logger.exception("Failed to decode base64 image chunk due to %s: %s", type(e).__name__, e)
            return None