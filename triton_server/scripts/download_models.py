import torch
from huggingface_hub import login
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from config import EMBEDDING_MODEL, RERANKER_MODEL, HF_TOKEN

login(HF_TOKEN)

print("=" * 60)
print(f"Downloading {EMBEDDING_MODEL} (encoder)...")
print("=" * 60)
AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
AutoProcessor.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
print("Done.\n")

print("=" * 60)
print(f"Downloading {RERANKER_MODEL} (reranker, 4-bit)...")
print("=" * 60)
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=True,
)
AutoModel.from_pretrained(
    RERANKER_MODEL,
    trust_remote_code=True,
    quantization_config=quant,
    device_map="auto",
)
AutoTokenizer.from_pretrained(RERANKER_MODEL, trust_remote_code=True)
print("Done.\n")

print("All models downloaded. You can now start the Triton server.")
