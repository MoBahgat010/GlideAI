"""
cross_encoder/1/model.py — Dynamic Request Unloading & Offset-Batched GPU Reranker.

Unloads candidate lists and queries from all incoming Triton batch requests first.
Constructs a single flat list of (query, candidate) pairs across requests,
executes a single batched GPU forward pass, and maps scores back using an offset list.
"""
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

RERANKER_MODEL = "jinaai/jina-reranker-m0"


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
        self._logger.log_info(f"[{RERANKER_MODEL}] Ready on device {device}.")

    def execute(self, requests: list) -> list:
        """
        1. Unload all requests first into a single combined pairs list.
        2. Keep an offset list mapping request indices to pair slice boundaries.
        3. Execute ONE batched forward pass on GPU for all candidate pairs.
        4. Slice the scores using the offset list and return per-request responses.
        """
        n_requests = len(requests)
        results = [None] * n_requests

        all_pairs = []
        request_offsets = []  # List of (req_idx, start_idx, end_idx)

        # Step 1: Unload all requests into combined flat pairs list
        for i, request in enumerate(requests):
            try:
                query_arr = pb_utils.get_input_tensor_by_name(request, "QUERY").as_numpy()
                candidates_arr = pb_utils.get_input_tensor_by_name(request, "CANDIDATES").as_numpy()

                query = query_arr.flat[0].decode("utf-8") if isinstance(query_arr.flat[0], bytes) else str(query_arr.flat[0])
                candidates = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in candidates_arr.flat]

                start_idx = len(all_pairs)
                for cand in candidates:
                    all_pairs.append([query, cand])
                end_idx = len(all_pairs)

                request_offsets.append((i, start_idx, end_idx))
            except Exception as exc:
                self._logger.log_error(f"[cross_encoder] Unload error request {i}: {exc}")
                results[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))

        # Step 2: Parallel forward pass for all pairs across requests
        if all_pairs:
            all_scores = self._score_pairs(all_pairs)

            # Step 3: Map scores back using request boundary offsets
            for req_idx, start_idx, end_idx in request_offsets:
                req_scores = all_scores[start_idx:end_idx]
                out_tensor = pb_utils.Tensor("SCORES", np.array(req_scores, dtype=np.float32))
                results[req_idx] = pb_utils.InferenceResponse(output_tensors=[out_tensor])

        return results

    def finalize(self) -> None:
        del self._model
        del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
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
