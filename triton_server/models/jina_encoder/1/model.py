import base64
import io
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoProcessor


class TritonPythonModel:
    """
    Triton Python backend for jinaai/jina-clip-v2 supporting both Text and Image inputs.

    Inputs:
      - TEXT:         bytes[batch, 1] (optional) UTF-8 text strings
      - IMAGE_BASE64: bytes[batch, 1] (optional) Base64 encoded image strings

    Output:
      - EMBEDDING:    float32[batch, 1024] L2-normalized multimodal dense vector
    """

    MODEL_NAME = "jinaai/jina-clip-v2"
    MAX_SEQ_LEN = 512

    def initialize(self, args: dict) -> None:
        self._logger = pb_utils.Logger
        self._logger.log_info(f"[jina_encoder] Loading {self.MODEL_NAME}...")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = (
            AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
            .to(self._device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )

        self._warmup()
        self._logger.log_info(
            f"[jina_encoder] Ready on {self._device} (dim={self._probe_dim()})"
        )

    # ------------------------------------------------------------------
    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BASE64")

            if text_tensor is not None:
                raw = text_tensor.as_numpy()
                texts = [row[0].decode("utf-8") for row in raw]
                safe_texts = [t if (t and t.strip()) else " " for t in texts]
                embeddings = self._encode_text(safe_texts)
            elif image_tensor is not None:
                raw = image_tensor.as_numpy()
                b64_strings = [row[0].decode("utf-8") for row in raw]
                images = [self._decode_b64(b) for b in b64_strings]
                embeddings = self._encode_image(images)
            else:
                out_err = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Neither TEXT nor IMAGE_BASE64 input tensor provided")
                )
                responses.append(out_err)
                continue

            out = pb_utils.Tensor("EMBEDDING", embeddings)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        del self._model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_SEQ_LEN,
        ).to(self._device)

        with torch.inference_mode():
            if hasattr(self._model, "get_text_features"):
                features = self._model.get_text_features(**inputs)
            else:
                out = self._model(**inputs)
                features = self._extract_features(out, is_image=False)

            normalized = F.normalize(features, dim=-1)

        return normalized.cpu().float().numpy()

    # ------------------------------------------------------------------
    def _encode_image(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(
            images=images,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(**inputs)
            else:
                out = self._model(**inputs)
                features = self._extract_features(out, is_image=True)

            normalized = F.normalize(features, dim=-1)

        return normalized.cpu().float().numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(output, is_image: bool = False) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        attr = "image_embeds" if is_image else "text_embeds"
        val = getattr(output, attr, None)
        if val is not None:
            return val
        val = getattr(output, "pooler_output", None)
        if val is not None:
            return val
        return output.last_hidden_state[:, 0, :]

    @staticmethod
    def _decode_b64(b64_str: str) -> Image.Image:
        if b64_str.startswith("data:") and ";base64," in b64_str:
            b64_str = b64_str.split(";base64,", 1)[1]
        try:
            raw = base64.b64decode(b64_str, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _probe_dim(self) -> int:
        sample = self._encode_text(["dim probe"])
        return sample.shape[-1]

    def _warmup(self) -> None:
        try:
            self._encode_text(["warmup"])
        except Exception:
            pass
