import io
import base64
import contextlib
import numpy as np
import torch
from PIL import Image
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoProcessor

EMBEDDING_MODEL = "google/siglip-large-patch16-384"


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        self._logger = pb_utils.Logger
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._on_cuda = self._device == "cuda"
        self._logger.log_info(f"[bi_encoder_ingestion] Loading {EMBEDDING_MODEL} on device {self._device}")

        self._model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            device_map=self._device,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )

        self._logger.log_info(f"[bi_encoder_ingestion] Ready on {self._device}.")

    def execute(self, requests: list) -> list:
        n_requests = len(requests)
        results = [None] * n_requests

        flat_texts = []
        text_offsets = []

        flat_images = []
        image_offsets = []

        for i, request in enumerate(requests):
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BASE64")

                has_input = False

                if text_tensor is not None:
                    raw = text_tensor.as_numpy()
                    texts = [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in raw.flat]
                    texts = [t if (t and t.strip()) else " " for t in texts]

                    start_idx = len(flat_texts)
                    flat_texts.extend(texts)
                    end_idx = len(flat_texts)
                    text_offsets.append((i, start_idx, end_idx))
                    has_input = True

                if image_tensor is not None:
                    raw = image_tensor.as_numpy()
                    b64s = [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in raw.flat]
                    images = [self._decode_b64(b) for b in b64s]

                    start_idx = len(flat_images)
                    flat_images.extend(images)
                    end_idx = len(flat_images)
                    image_offsets.append((i, start_idx, end_idx))
                    has_input = True

                if not has_input:
                    results[i] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Request provided neither TEXT nor IMAGE_BASE64 input.")
                    )
            except Exception as exc:
                self._logger.log_error(f"[bi_encoder_ingestion] Unload error on request {i}: {exc}")
                results[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))

        text_embeddings = None
        image_embeddings = None

        if self._on_cuda and flat_texts and flat_images:
            stream_text = torch.cuda.Stream()
            stream_image = torch.cuda.Stream()

            with torch.cuda.stream(stream_text):
                text_embeddings = self._encode_text(flat_texts)

            with torch.cuda.stream(stream_image):
                image_embeddings = self._encode_image(flat_images)

            torch.cuda.synchronize()
        else:
            if flat_texts:
                text_embeddings = self._encode_text(flat_texts)
            if flat_images:
                image_embeddings = self._encode_image(flat_images)

        for i in range(n_requests):
            if results[i] is not None and results[i].has_error():
                continue

            req_embeddings = []

            for req_idx, start_idx, end_idx in text_offsets:
                if req_idx == i and text_embeddings is not None:
                    req_embeddings.extend(text_embeddings[start_idx:end_idx].tolist())

            for req_idx, start_idx, end_idx in image_offsets:
                if req_idx == i and image_embeddings is not None:
                    req_embeddings.extend(image_embeddings[start_idx:end_idx].tolist())

            if req_embeddings:
                out_tensor = pb_utils.Tensor("EMBEDDING", np.array(req_embeddings, dtype=np.float32))
                results[i] = pb_utils.InferenceResponse(output_tensors=[out_tensor])

        torch.cuda.empty_cache()
        return results

    def finalize(self) -> None:
        del self._model
        del self._processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_b64(self, b64_str: str) -> Image.Image:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self._device)
        text_embeds = self._model.get_text_features(**inputs)
        if hasattr(text_embeds, "pooler_output") and text_embeds.pooler_output is not None:
            text_embeds = text_embeds.pooler_output
        elif hasattr(text_embeds, "last_hidden_state") and text_embeds.last_hidden_state is not None:
            text_embeds = text_embeds.last_hidden_state[:, 0]
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds.cpu().numpy()

    @torch.no_grad()
    def _encode_image(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        image_embeds = self._model.get_image_features(**inputs)
        if hasattr(image_embeds, "pooler_output") and image_embeds.pooler_output is not None:
            image_embeds = image_embeds.pooler_output
        elif hasattr(image_embeds, "last_hidden_state") and image_embeds.last_hidden_state is not None:
            image_embeds = image_embeds.last_hidden_state[:, 0]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds.cpu().numpy()
