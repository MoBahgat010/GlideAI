"""
MultimodalEncoder: Bi-encoder client for text and image embeddings via Triton Server gRPC.

Target Triton Models:
  - Ingestion embeddings  → bi_encoder_ingestion
  - Query embeddings      → bi_encoder_retrieval

NOTE on dtype:
  tritonclient maps numpy dtype=object → Triton "BYTES".
  All config.pbtxt inputs declare TYPE_BYTES, and InferInput is constructed
  with datatype="BYTES".  Items must be plain Python str objects.

NOTE on batching:
  When a document batch contains both text and image chunks, each input type
  is sent in a separate infer call (text-only, image-only) because Triton
  requires all inputs within a single request to share the same batch size.
"""
import logging
from typing import List

import numpy as np
from langchain_core.documents import Document
import tritonclient.grpc as grpcclient

from config import TRITON_GRPC_URL

logger = logging.getLogger("ingestion.embedding")

class MultimodalEncoder:
    def __init__(self, url: str = TRITON_GRPC_URL, d_model: int = 768):
        self._url = url
        self.d_model = d_model
        self._client = grpcclient.InferenceServerClient(url=self._url)
        logger.info("MultimodalEncoder initialized for Triton gRPC at %s", url)

    @staticmethod
    def _str_tensor(name: str, items: List[str]) -> grpcclient.InferInput:
        """Build a BYTES InferInput from a list of plain strings.

        With max_batch_size > 0, Triton prepends the batch dim automatically,
        so dims: [-1] in config means the full tensor shape is [batch, -1].
        We send shape (N, 1) — N items, each a 1-element BYTES array.
        """
        arr = np.array(items, dtype=object).reshape(-1, 1)
        inp = grpcclient.InferInput(name, arr.shape, "BYTES")
        inp.set_data_from_numpy(arr)
        return inp

    def _infer(self, inputs: list) -> np.ndarray:
        response = self._client.infer(
            model_name="bi_encoder_ingestion",
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("EMBEDDING")],
        )
        return response.as_numpy("EMBEDDING")

    def _infer_batched(self, tensor_name: str, items: List[str]) -> List[List[float]]:
        results = []
        arr = self._infer([self._str_tensor(tensor_name, items)])
        if arr is None:
            results.extend([[0.0] * self.d_model] * len(items))
        else:
            results.extend(arr.tolist())
        return results

    def embed_chunks(self, chunks: List[Document]) -> List[List[float]]:
        """
        Embed document chunks for vector indexing.
        Text and image chunks are sent in separate Triton requests to satisfy
        Triton's requirement that all inputs share the same batch size.
        """
        if not chunks:
            return []

        text_batch = []
        image_batch = []
        text_indices = []
        image_indices = []

        for idx, chunk in enumerate(chunks):
            if chunk.metadata.get("type") == "image":
                image_batch.append(str(chunk.page_content))
                image_indices.append(idx)
            else:
                text_batch.append(str(chunk.page_content))
                text_indices.append(idx)

        final_embeddings: List = [None] * len(chunks)

        if text_batch:
            text_vecs = self._infer_batched("TEXT", text_batch)
            for i, chunk_idx in enumerate(text_indices):
                final_embeddings[chunk_idx] = text_vecs[i]

        if image_batch:
            image_vecs = self._infer_batched("IMAGE_BASE64", image_batch)
            for i, chunk_idx in enumerate(image_indices):
                final_embeddings[chunk_idx] = image_vecs[i]

        return final_embeddings

    def encode_query(self, query: str) -> List[float]:
        """Encode a single search query string using bi_encoder_retrieval."""
        response = self._client.infer(
            model_name="bi_encoder_retrieval",
            inputs=[self._str_tensor("TEXT", [str(query)])],
            outputs=[grpcclient.InferRequestedOutput("EMBEDDING")],
        )
        result = response.as_numpy("EMBEDDING")
        if result is not None and len(result) > 0:
            return result[0].tolist()
        return [0.0] * self.d_model