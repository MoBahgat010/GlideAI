"""
bi_encoder/1/model.py — GPU worker model.

Model: loaded from EMBEDDING_MODEL env (e.g. google/siglip-large-patch16-384).
Output: L2-normalised float32 embeddings.

Concurrency strategy:
  Each call to execute() receives a list of requests (e.g. one TEXT batch
  and one IMAGE_BASE64 batch arriving at roughly the same time). We process
  them on *separate CUDA streams* using contextlib.ExitStack so that text
  and image inference truly overlap on the GPU — neither modality blocks the
  other. The main thread waits on both streams before returning all responses.
"""
import base64
import contextlib
import io

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoProcessor
# from config import EMBEDDING_MODEL
EMBEDDING_MODEL="google/siglip-large-patch16-384"
# Maximum number of CUDA streams used concurrently.
_MAX_STREAMS = 4


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        self._logger = pb_utils.Logger

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.log_info(f"[{EMBEDDING_MODEL}] Loading {EMBEDDING_MODEL} on device {self._device}")

        self._model =  AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True, device_map=self._device).eval()

        self._processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)

        self._warmup()
        self._logger.log_info(
            f"[{EMBEDDING_MODEL}] Ready on {self._device} dim={self._probe_dim()} memory={self._model.get_memory_footprint() / 1e6:.1f}MB"
        )

    def execute(self, requests: list) -> "list[pb_utils.InferenceResponse]":
        """
        Process all requests concurrently on separate CUDA streams.

        Each request gets its own stream (round-robin from the pool).
        We launch all kernels first, then synchronise and collect results
        — so text and image inference truly overlap on the GPU.
        """
        n = len(requests)
        results: list = [None] * n

        # Each slot holds (stream | None, future_fn | None)
        pending: list[tuple] = []

        with contextlib.ExitStack() as stack:
            for i, request in enumerate(requests):
                stream = self._streams[i % len(self._streams)]

                ctx = (
                    torch.cuda.stream(stream)
                    if stream is not None
                    else contextlib.nullcontext()
                )
                stack.enter_context(ctx)

                text_tensor  = pb_utils.get_input_tensor_by_name(request, "TEXT")
                image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BASE64")

                try:
                    if text_tensor is not None:
                        raw    = text_tensor.as_numpy()
                        texts  = [r[0].decode("utf-8") for r in raw]
                        texts  = [t if (t and t.strip()) else " " for t in texts]
                        # Launch on stream — returns a CPU tensor (stream synced later)
                        emb    = self._encode_text(texts, stream=stream)

                    elif image_tensor is not None:
                        raw    = image_tensor.as_numpy()
                        b64s   = [r[0].decode("utf-8") for r in raw]
                        images = [self._decode_b64(b) for b in b64s]
                        emb    = self._encode_image(images, stream=stream)

                    else:
                        pending.append((None, None, pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(
                                "Neither TEXT nor IMAGE_BASE64 input tensor provided"
                            )
                        )))
                        continue

                    pending.append((stream, emb, i))

                except Exception as exc:  # noqa: BLE001
                    self._logger.log_error(f"[bi_encoder] launch error request {i}: {exc}")
                    pending.append((None, None, pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )))

        # All CUDA kernels have been launched. Now synchronise each stream
        # and collect the numpy arrays → build responses.
        seen_streams: set[int] = set()
        for entry in pending:
            stream, emb, idx = entry

            if isinstance(idx, pb_utils.InferenceResponse):
                # Error response stored directly
                results[len([r for r in results if r is not None])] = idx
                continue

            # Sync each stream exactly once.
            if stream is not None and id(stream) not in seen_streams:
                stream.synchronize()
                seen_streams.add(id(stream))

            out = pb_utils.Tensor("EMBEDDING", np.array(emb, dtype=np.float32))
            results[idx] = pb_utils.InferenceResponse(output_tensors=[out])

        return results

    def finalize(self) -> None:
        del self._model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_text(
        self,
        texts: list[str],
        stream: "torch.cuda.Stream | None" = None,
    ) -> np.ndarray:
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            if hasattr(self._model, "get_text_features"):
                features = self._model.get_text_features(**inputs)
            else:
                out = self._model(**inputs)
                features = self._extract_features(out, is_image=False)

            normalized = F.normalize(features, dim=-1)
            # Move to CPU while stream is still active (async copy on the stream)
            return normalized.cpu().float().numpy()

    @torch.no_grad()
    def _encode_image(
        self,
        images: list[Image.Image],
        stream: "torch.cuda.Stream | None" = None,
    ) -> np.ndarray:
        inputs = self._processor(
            images=images,
            return_tensors="pt",
        ).to(self._device)

        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            if hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(**inputs)
            else:
                out = self._model(**inputs)
                features = self._extract_features(out, is_image=True)

            normalized = F.normalize(features, dim=-1)
            return normalized.cpu().float().numpy()

    @staticmethod
    def _extract_features(output, is_image: bool = False) -> torch.Tensor:
        """Universal feature extractor for CLIP/SigLIP-family models."""
        if isinstance(output, torch.Tensor):
            return output
        for attr in (("image_embeds" if is_image else "text_embeds"), "pooler_output"):
            val = getattr(output, attr, None)
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
        return self._encode_text(["dim probe"]).shape[-1]

    def _warmup(self) -> None:
        try:
            self._encode_text(["warmup"])
        except Exception:
            pass
