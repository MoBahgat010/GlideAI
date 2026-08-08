import base64
import binascii
import io
import logging
from typing import Any

import torch
from torch.nn import functional as F
from PIL import Image, UnidentifiedImageError
from langchain_core.documents import Document
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("ingestion.embedding")

class MultimodalEncoder:
    def __init__(self, device: str, batch_size: int, model_name: str):
        self.batch_size = batch_size

        self.model_name = model_name
        self.device = device
        
        logger.info("Initializing MultimodalEncoder with model_name=%s on device=%s", model_name, device)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        sample_emb = self.encode_text(["dimension check"])[0]
        self.d_model = len(sample_emb)
        logger.info("Loaded %s (dim=%s) on device=%s", model_name, self.d_model, device)

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [t if (t and t.strip()) else " " for t in texts]
        inputs = self.processor(text=safe_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                raw = self.model.get_text_features(**inputs)
            else:
                raw = self.model(**inputs)
            features = self._extract_features(raw)
            normalized = F.normalize(features, dim=-1)
            return normalized.cpu().tolist()

    def encode_image(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                raw = self.model.get_image_features(**inputs)
            else:
                raw = self.model(**inputs)
            features = self._extract_features(raw)
            normalized = F.normalize(features, dim=-1)
            return normalized.cpu().tolist()

    def embed_chunks(self, chunks: list[Document]) -> list[dict[str, Any]]:
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
                    logger.warning("Skipping image chunk at index %d: could not decode image data", i)
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
    def _extract_features(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            val = getattr(output, attr, None)
            if val is not None:
                return val

        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :]

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