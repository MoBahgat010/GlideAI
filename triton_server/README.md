# Triton Inference Server — Setup & Usage

## What this is

The `triton_server/` directory contains everything needed to run NVIDIA Triton
Inference Server hosting your two local GPU models:

| Triton model | Source model | Purpose |
|---|---|---|
| `jina_encoder` | `jinaai/jina-clip-v2` | Text embedding for query vectors |
| `jina_reranker` | `jinaai/jina-reranker-m0` (4-bit) | Cross-encoder reranking |

**Why Triton instead of running models in-process?**

When models run inside the FastAPI process, each inference call blocks a
thread.  Even with `asyncio.to_thread`, you are limited to N concurrent GPU
calls (where N must be small to avoid OOM) and you have no dynamic batching.

Triton solves both problems:

1. **Dynamic batching (encoder)** — Triton collects individual embedding
   requests that arrive within a configurable time window (10 ms by default)
   and fuses them into a single GPU kernel call.  32 users asking 32 questions
   simultaneously produces **one** GPU forward pass instead of 32.

2. **Queue management (reranker)** — Each reranking call already processes a
   batch of ~50 candidate pairs.  Triton queues concurrent rerank requests
   and serialises them to the GPU without OOM.

3. **FastAPI is GPU-free** — The FastAPI process only handles HTTP routing,
   Redis, and MongoDB.  All GPU compute lives in the Triton container.
   The 1-core CPU constraint no longer matters for inference latency.

4. **Metrics** — Triton exposes Prometheus metrics on port 8002
   (throughput, queue depth, latency percentiles per model).

---

## Directory layout

```
triton_server/
├── Dockerfile                          extends nvcr tritonserver + deps
├── docker-compose.yml
├── scripts/
│   └── download_models.py              pre-download HF weights to cache
└── models/                             Triton model repository
    ├── jina_encoder/
    │   ├── config.pbtxt                model config (dynamic batching on)
    │   └── 1/
    │       └── model.py                Python backend (loads jina-clip-v2)
    └── jina_reranker/
        ├── config.pbtxt                model config (batching off)
        └── 1/
            └── model.py                Python backend (loads reranker 4-bit)
```

---

## Step-by-step usage

### 1. Pre-download model weights (run once on the host)

```bash
cd /path/to/Enterprise_RAG
python triton_server/scripts/download_models.py
```

This populates `~/.cache/huggingface`.  The Docker container mounts the same
directory so no downloads happen at container start.

### 2. Build the image

```bash
cd triton_server
docker compose build
```

The `Dockerfile` extends the official NVIDIA Triton image and installs
`transformers`, `bitsandbytes`, `accelerate`, etc. into the container's Python.

### 3. Start Triton

```bash
docker compose up -d
```

Both models load when the container starts (~60-90 s for the reranker).
Monitor progress:

```bash
docker compose logs -f triton
```

Wait for:
```
[jina_encoder] Ready on cuda (dim=1024)
[jina_reranker] Ready.
```

### 4. Enable in the FastAPI server

Add to your `.env`:

```env
USE_TRITON=true
TRITON_URL=localhost:8000
```

When the main refactor plan is implemented, the `RAGContainer.init()` method
will read `USE_TRITON` and swap the local models for `TritonEncoder` and
`TritonReranker` from `server/src/services/triton_client.py`.

When `USE_TRITON=true`:
- The FastAPI process loads **no GPU models at all**
- No semaphores needed (Triton handles queueing internally)

### 5. Verify Triton is healthy

```bash
# Model list
curl http://localhost:8000/v2/models

# Encoder ready?
curl http://localhost:8000/v2/models/jina_encoder/ready

# Prometheus metrics
curl http://localhost:8002/metrics | grep nv_inference
```

---

## How the Python backends work

### `jina_encoder/1/model.py`

Triton's Python backend calls `initialize()` once at startup and `execute()`
for every inference request.  With `dynamic_batching` enabled, Triton
accumulates requests arriving within `max_queue_delay_microseconds` (10 ms)
and passes them as a single merged request with batch_size > 1.

So if 8 users submit queries within 10 ms, `execute()` receives one
request with TEXT tensor of shape `(8, 1)`.  The model does a single
forward pass for all 8 queries.

### `jina_reranker/1/model.py`

Dynamic batching is **off** (`max_batch_size: 0`) because each reranking
request is already a batch: (query, passage_1), (query, passage_2), ...
for all ~50 candidates.  Merging two users' batches would help GPU utilisation
slightly but doubles latency for the first user — sequential queuing is safer.

---

## Triton client (`server/src/services/triton_client.py`)

| Class | Replaces | Same method signatures |
|---|---|---|
| `TritonEncoder` | `MultimodalEncoder` | `encode_text()`, `embed_chunks()` |
| `TritonReranker` | `HybridReranker` | `rerank()` |

---

## Stopping

```bash
cd triton_server
docker compose down
```

---

## Tuning dynamic batching

Edit `triton_server/models/jina_encoder/config.pbtxt`:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 10000   # increase for higher throughput
}
```

Lower value → lower latency, smaller batches.  
Higher value → larger batches, better GPU utilisation, slightly higher latency.

10 ms is a good starting point. Profile with `nv_inference_queue_duration_us`
in the Prometheus metrics to tune.
