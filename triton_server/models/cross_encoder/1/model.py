"""
cross_encoder/1/model.py — GPU reranker model.

Receives:
  QUERY      — shape [1],    single query string.
  CANDIDATES — shape [-1],   array of candidate strings (text or image captions).

Returns:
  SCORES — shape [-1], float32 relevance scores, one per candidate.

Multiple simultaneous requests are dispatched on separate CUDA streams
via contextlib.ExitStack so they overlap on the GPU rather than serialising.
"""
import contextlib

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
# from config import RERANKER_MODEL
RERANKER_MODEL="jinaai/jina-reranker-m0"
_MAX_STREAMS = 4


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        self._logger = pb_utils.Logger
        self._logger.log_info(f"[{RERANKER_MODEL}] Loading {RERANKER_MODEL} (4-bit)...")

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
        )
        self._model = AutoModel.from_pretrained(
            RERANKER_MODEL,
            trust_remote_code=True,
            quantization_config=quant_cfg,
            device_map="auto",
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, trust_remote_code=True)

        device = next(self._model.parameters()).device
        self._on_cuda = device.type == "cuda"

        if self._on_cuda:
            self._streams = [torch.cuda.Stream() for _ in range(_MAX_STREAMS)]
        else:
            self._streams = [None] * _MAX_STREAMS

        self._logger.log_info(f"[{RERANKER_MODEL}] ready on {device}.")

    def execute(self, requests: list) -> "list[pb_utils.InferenceResponse]":
        """
        Launch all scoring jobs on separate CUDA streams, then synchronise
        and return — concurrent requests overlap on the GPU.
        """
        n = len(requests)
        results: list = [None] * n
        pending: list[tuple] = []  # (stream, scores_array, request_index)

        with contextlib.ExitStack() as stack:
            for i, request in enumerate(requests):
                stream = self._streams[i % len(self._streams)]

                ctx = (
                    torch.cuda.stream(stream)
                    if stream is not None
                    else contextlib.nullcontext()
                )
                stack.enter_context(ctx)

                try:
                    query_arr = pb_utils.get_input_tensor_by_name(request, "QUERY").as_numpy()
                    candidates_arr = pb_utils.get_input_tensor_by_name(request, "CANDIDATES").as_numpy()

                    query = query_arr.flat[0].decode("utf-8")
                    candidates = [p.decode("utf-8") for p in candidates_arr.flat]

                    scores = self._score(query, candidates, stream=stream)
                    pending.append((stream, scores, i))

                except Exception as exc:  # noqa: BLE001
                    self._logger.log_error(f"[cross_encoder] launch error request {i}: {exc}")
                    results[i] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )

        # Synchronise each stream once, then build responses.
        seen_streams: set[int] = set()
        for stream, scores, idx in pending:
            if stream is not None and id(stream) not in seen_streams:
                stream.synchronize()
                seen_streams.add(id(stream))

            out = pb_utils.Tensor("SCORES", np.array(scores, dtype=np.float32))
            results[idx] = pb_utils.InferenceResponse(output_tensors=[out])

        return results

    def finalize(self) -> None:
        del self._model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _score(
        self,
        query: str,
        candidates: list[str],
        stream: "torch.cuda.Stream | None" = None,
    ) -> list[float]:
        pairs = [[query, c] for c in candidates]

        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            if hasattr(self._model, "compute_score"):
                raw = self._model.compute_score(pairs)
            else:
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                raw = outputs.logits.squeeze(-1).cpu().float().numpy().tolist()

        if isinstance(raw, (float, int)):
            raw = [float(raw)]

        return [float(s) for s in raw]
