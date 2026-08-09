"""
Pre-download both model weights into the HuggingFace cache before starting
Triton.  The Triton container mounts the same cache directory, so models are
immediately available without any network call at runtime.

Run once on the host (NOT inside the container):
    python triton_server/scripts/download_models.py
"""
import torch
from huggingface_hub import login
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
import os

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(HF_TOKEN)

print("=" * 60)
print("Downloading jinaai/jina-clip-v2 (encoder)...")
print("=" * 60)
AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
print("Done.\n")

print("=" * 60)
print("Downloading jinaai/jina-reranker-m0 (reranker, 4-bit)...")
print("=" * 60)
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=True,
)
AutoModel.from_pretrained(
    "jinaai/jina-reranker-m0",
    trust_remote_code=True,
    quantization_config=quant,
    device_map="auto",
)
AutoTokenizer.from_pretrained("jinaai/jina-reranker-m0", trust_remote_code=True)
print("Done.\n")

print("All models downloaded. You can now start the Triton server.")
