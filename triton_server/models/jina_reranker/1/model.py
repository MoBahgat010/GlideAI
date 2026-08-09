import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class TritonPythonModel:
    """
    Triton Python backend for jinaai/jina-reranker-m0 (4-bit quantised).

    Inputs:
      QUERY    — bytes[1]    the search query (single string)
      PASSAGES — bytes[-1]   N passage strings to score against the query

    Output:
      SCORES   — float32[-1] relevance score per passage (higher = more relevant)

    Dynamic batching is OFF (max_batch_size: 0) because each request is
    already a variable-length batch of (query, passage) pairs.  Multiple
    concurrent reranking requests queue up and are processed in arrival order,
    which is correct since each call already saturates the GPU.
    """

    MODEL_NAME = "jinaai/jina-reranker-m0"

    def initialize(self, args: dict) -> None:
        self._logger = pb_utils.Logger
        self._logger.log_info(f"[jina_reranker] Loading {self.MODEL_NAME} (4-bit)...")

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
        )
        self._model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            quantization_config=quant_cfg,
            device_map="auto",
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )
        self._logger.log_info("[jina_reranker] Ready.")

    # ------------------------------------------------------------------
    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            query_arr = pb_utils.get_input_tensor_by_name(request, "QUERY").as_numpy()
            passages_arr = pb_utils.get_input_tensor_by_name(request, "PASSAGES").as_numpy()

            query = query_arr.flat[0].decode("utf-8")
            passages = [p.decode("utf-8") for p in passages_arr.flat]

            scores = self._score(query, passages)

            out = pb_utils.Tensor("SCORES", np.array(scores, dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        del self._model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [[query, p] for p in passages]

        with torch.inference_mode():
            if hasattr(self._model, "compute_score"):
                scores = self._model.compute_score(pairs)
            else:
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().float().numpy().tolist()

        if isinstance(scores, (float, int)):
            scores = [float(scores)]

        return [float(s) for s in scores]
