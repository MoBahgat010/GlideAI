"""
Multimodal encoder using jinaai/jina-clip-v2.

Dual-tower architecture:
  - Text tower  → model.get_text_features()  → 1024-dim normalised vector
  - Image tower → model.get_image_features() → 1024-dim normalised vector

Both towers share the same embedding space, so text and image vectors
are directly comparable via cosine similarity in Pinecone.

Also provides a LangChain Embeddings adapter (JinaClipTextEmbeddings)
for use with SemanticChunker.
"""

import logging
import threading
from typing import Any

import torch
from torch import nn
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("ingestion.embedding")


class MultimodalEncoder(nn.Module):
    """
    Dual-tower encoder backed by jinaai/jina-clip-v2.

    All parameters are frozen (inference-only).
    Supports batched encoding for both text and images.
    """

    def __init__(self, device: str | None = None, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model
        requested_device = (device or "").strip().lower() if device else ""
        if requested_device in {"cuda", "cuda:0", "cpu"}:
            self.device = requested_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA was requested but is not available; falling back to CPU."
            )
            self.device = "cpu"
        self._model_lock = threading.Lock()

        # -----------------------------------------------------------------------------

        logger.info("Loading jinaai/jina-clip-v2 on device=%s …", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2", 
            trust_remote_code=True,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.parameters())
        logger.info("jina-clip-v2 loaded — %s parameters (all frozen)", f"{num_params:,}")

        for param in self.parameters():
            param.requires_grad = False

    # ── public API ────────────────────────────────────────────────────────────

    def get_dimension(self) -> int:
        """Return the embedding dimension (1024)."""
        return self.d_model

    @torch.no_grad()
    def encode_text(
        self, texts: list[str], batch_size: int = 4
    ) -> list[list[float]]:
        """
        Encode a list of texts via the text tower.

        Returns a list of 1024-dim normalised vectors (as plain Python lists).
        Processes in small batches to manage VRAM/RAM (jina-clip-v2 supports up to 8192 context length, which can cause OOM with large batches).
        """
        safe_texts = [t if (t and t.strip()) else " " for t in texts]
        all_embeddings: list[list[float]] = []

        for start in range(0, len(safe_texts), batch_size):
            batch = safe_texts[start : start + batch_size]
            logger.info(
                "Encoding text batch %d-%d/%d (batch_size=%d)",
                start + 1,
                min(start + len(batch), len(safe_texts)),
                len(safe_texts),
                len(batch),
            )
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with self._model_lock:
                features = self.model.get_text_features(**tokens).float()
            features = nn.functional.normalize(features, p=2, dim=-1, eps=1e-12)
            if features.ndim != 2 or features.shape[-1] != self.d_model:
                raise ValueError(
                    f"Unexpected text embedding shape {tuple(features.shape)} "
                    f"(expected (*, {self.d_model}))"
                )
            if not torch.isfinite(features).all():
                raise ValueError(
                    "Text encoder produced non-finite values after normalization. "
                    "This usually means the model returned an all-zero or invalid feature vector."
                )
            logger.info(
                "Text embedding batch output shape=%s dtype=%s device=%s",
                tuple(features.shape),
                features.dtype,
                features.device,
            )
            all_embeddings.extend(features.cpu().tolist())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.debug(
            "encode_text: %d texts → %d vectors (dim=%d)",
            len(texts), len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    @torch.no_grad()
    def encode_image(
        self, images: list[Image.Image], batch_size: int = 2
    ) -> list[list[float]]:
        """
        Encode a list of PIL Images via the image tower.

        Returns a list of 1024-dim normalised vectors (as plain Python lists).
        Uses a very small default batch_size because images consume high VRAM.
        """
        all_embeddings: list[list[float]] = []

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            logger.info(
                "Encoding image batch %d-%d/%d (batch_size=%d)",
                start + 1,
                min(start + len(batch), len(images)),
                len(images),
                len(batch),
            )
            inputs = self.processor(
                images=batch, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._model_lock:
                features = self.model.get_image_features(**inputs).float()
            features = nn.functional.normalize(features, p=2, dim=-1, eps=1e-12)
            if features.ndim != 2 or features.shape[-1] != self.d_model:
                raise ValueError(
                    f"Unexpected image embedding shape {tuple(features.shape)} "
                    f"(expected (*, {self.d_model}))"
                )
            if not torch.isfinite(features).all():
                raise ValueError(
                    "Image encoder produced non-finite values after normalization. "
                    "This usually means the model returned an all-zero or invalid feature vector."
                )
            logger.info(
                "Image embedding batch output shape=%s dtype=%s device=%s",
                tuple(features.shape),
                features.dtype,
                features.device,
            )
            all_embeddings.extend(features.cpu().tolist())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.debug(
            "encode_image: %d images → %d vectors (dim=%d)",
            len(images), len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Use encode_text() or encode_image() directly."
        )


class JinaClipTextEmbeddings(Embeddings):
    """
    LangChain Embeddings adapter wrapping MultimodalEncoder.encode_text().

    Used by SemanticChunker for computing inter-sentence similarity
    during chunk-boundary detection.
    """

    def __init__(self, encoder: MultimodalEncoder):
        self._encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode_text(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encoder.encode_text([text])[0]
