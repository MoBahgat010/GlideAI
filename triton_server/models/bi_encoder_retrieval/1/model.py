import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoProcessor

EMBEDDING_MODEL = "google/siglip-large-patch16-384"


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        self._logger = pb_utils.Logger
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.log_info(f"[bi_encoder_retrieval] Loading {EMBEDDING_MODEL} on device {self._device}")

        self._model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            device_map=self._device,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )

        self._logger.log_info(f"[bi_encoder_retrieval] Ready on {self._device}.")

    def execute(self, requests: list) -> list:
        """
        Unload query text requests across the batch into a single flat array.
        Execute a single batched GPU text embedding pass, then return EMBEDDING outputs.
        """
        n_requests = len(requests)
        results = [None] * n_requests

        flat_queries = []
        request_offsets = []
        for i, request in enumerate(requests):
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                if text_tensor is not None:
                    raw = text_tensor.as_numpy()
                    queries = [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in raw.flat]
                    queries = [q if (q and q.strip()) else " " for q in queries]

                    start_idx = len(flat_queries)
                    flat_queries.extend(queries)
                    end_idx = len(flat_queries)
                    request_offsets.append((i, start_idx, end_idx))
                else:
                    results[i] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Request provided no TEXT input.")
                    )
            except Exception as exc:
                self._logger.log_error(f"[bi_encoder_retrieval] Unload error request {i}: {exc}")
                results[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))

        if flat_queries:
            embeddings = self._encode_text(flat_queries)
            for req_idx, start_idx, end_idx in request_offsets:
                req_emb = embeddings[start_idx:end_idx]
                out_tensor = pb_utils.Tensor("EMBEDDING", req_emb.astype(np.float32))
                results[req_idx] = pb_utils.InferenceResponse(output_tensors=[out_tensor])

        return results

    def finalize(self) -> None:
        del self._model
        del self._processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self._device)
        text_embeds = self._model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds.cpu().numpy()
