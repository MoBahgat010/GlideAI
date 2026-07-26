"""
app.py — RAG pipeline Gradio UI
  Tab 1 · Upload & Ingest  — chunked file upload → S3/disk → Celery job
                             with live streaming stage progress
  Tab 2 · Ask              — query rewrite + HyDE + retrieval + reranking
                             with streaming token answer
"""

import time
import uuid
from pathlib import Path

import gradio as gr
from celery.result import AsyncResult
from openai import OpenAI

import logging

import rag_config as cfg
from storage.object_storage import ObjectStorage
from storage.pinecone import PineconeVDB
from ingestion.embedding import MultimodalEncoder
from retrieval.pipeline import RetrievalPipeline
from tasks import celery_app, run_ingestion as celery_run_ingestion  # noqa: F401

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

storage = ObjectStorage(s3_uri=cfg.S3_URI, local_dir=cfg.UPLOAD_DIR)

_retrieval_pipeline = None

def get_retrieval_pipeline():
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        logger.info("Initializing heavy models (encoder & reranker) lazily...")
        
        # VERY IMPORTANT: We MUST initialize the reranker BEFORE the encoder.
        # Jina-clip-v2 (used by encoder) loads a custom `xlm-roberta-flash-implementation`
        # via `trust_remote_code=True` which overrides the global `transformers` registry for XLM-RoBERTa.
        # If the reranker (which uses standard XLM-RoBERTa) is loaded AFTER, it uses Jina's custom rotary code
        # and crashes with a tensor size mismatch! Loading the reranker first caches the correct implementation.
        from retrieval.reranker import HybridReranker
        pre_loaded_reranker = HybridReranker()
        
        encoder = MultimodalEncoder(device=cfg.EMBEDDING_DEVICE)
        vdb = PineconeVDB(cfg.PINECONE_API_KEY, cfg.PINECONE_INDEX_NAME, encoder.get_dimension())
        nvidia_client = OpenAI(api_key=cfg.NVIDIA_API_KEY, base_url=cfg.NVIDIA_BASE_URL)
        
        _retrieval_pipeline = RetrievalPipeline(
            encoder=encoder,
            vdb=vdb,
            nvidia_client=nvidia_client,
            light_model=cfg.LIGHT_WEIGHT_MODEL,
            heavy_model=cfg.HEAVY_WEIGHT_MODEL,
            retrieve_top_k=cfg.RETRIEVE_TOP_K,
            rerank_top_k=cfg.RERANK_TOP_K,
            reranker=pre_loaded_reranker,
        )
    return _retrieval_pipeline

# ── Stage decorations ──────────────────────────────────────────────────────────
ICONS = {
    "DOWNLOADING": "📥",
    "PARSING":     "📄",
    "CHUNKING":    "✂️",
    "EMBEDDING":   "🧠",
    "DONE":        "✅",
    "REWRITING":   "✏️",
    "RETRIEVING":  "🔍",
    "RERANKING":   "🎯",
    "ERROR":       "❌",
}


def _icon(stage: str) -> str:
    return ICONS.get(stage, "🔄")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Upload & Ingest
# ══════════════════════════════════════════════════════════════════════════════

def upload_and_ingest(files):
    """Generator: yields accumulated log strings so Gradio streams them live."""
    if not files:
        yield "⚠️  Please upload at least one file first."
        return

    log = ""
    job_id = str(uuid.uuid4())

    def append(line: str) -> str:
        nonlocal log
        log = log + line + "\n"
        return log

    yield append(f"📤  **Uploading {len(files)} file(s)…**")

    storage_keys: list[str] = []
    for f in files:
        key = Path(f).name
        with open(f, "rb") as fh:
            storage.upload(fh, key)
        storage_keys.append(key)
        yield append(f"   ✓  `{key}` uploaded")

    yield append(f"\n🚀  **Dispatching ingestion job** `{job_id[:8]}…`")
    task = celery_run_ingestion.delay(job_id, storage_keys)

    last_stage = ""
    poll_sleep = 1.5

    while True:
        result = AsyncResult(task.id, app=celery_app)

        if result.state == "PENDING":
            yield append("⏳  Waiting for Celery worker…")

        elif result.state == "STARTED":
            yield append("🔄  Worker started, loading models…")

        elif result.state == "PROGRESS":
            info = result.info or {}
            stage   = info.get("stage", "")
            message = info.get("message", "")
            pct     = int(info.get("pct", 0) * 100)
            if stage != last_stage:
                yield append(f"\n{_icon(stage)}  **{stage}** ({pct}%)  —  {message}")
                last_stage = stage

        elif result.state == "SUCCESS":
            info = result.result or {}
            yield append(
                f"\n✅  **Ingestion complete!**\n"
                f"    • Chunks indexed : **{info.get('chunks_indexed', '?')}**\n"
                f"    • Job ID         : `{job_id}`"
            )
            return

        elif result.state == "FAILURE":
            yield append(f"\n❌  **Task failed**: {result.info}")
            return

        time.sleep(poll_sleep)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Ask
# ══════════════════════════════════════════════════════════════════════════════

def ask_stream(query: str, history: list):
    """
    Generator for the chatbot:
      1. Show retrieval stage progress in the last message bubble (streamed live)
      2. Stream the final answer token by token
    """
    import queue
    import threading

    if not query.strip():
        yield history, "", "", ""
        return

    print(f"\n{'='*60}")
    print(f"[ASK] Query received: {query!r}")
    print(f"{'='*60}")

    history = list(history or [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": "⏳ Retrieving…"})
    yield history, "", "", ""

    stage_log: list[str] = []
    result_holder: list[dict] = []
    progress_queue: queue.Queue = queue.Queue()

    def on_progress(stage: str, msg: str):
        print(f"[PROGRESS] {stage}: {msg}")
        stage_log.append(f"{_icon(stage)} **{stage}** — {msg}")
        progress_queue.put(("progress", "\n".join(stage_log)))

    def run_retrieval():
        print("[RETRIEVAL] Thread started")
        try:
            pipeline = get_retrieval_pipeline()
            result = pipeline.retrieve(query, progress=on_progress)
            print(f"[RETRIEVAL] Done. {len(result.get('results', []))} results returned.")
            result_holder.append(result)
        except Exception as exc:
            import traceback
            print(f"[RETRIEVAL ERROR] {exc}")
            traceback.print_exc()
            result_holder.append({"error": str(exc)})
        finally:
            print("[RETRIEVAL] Thread finishing, sending 'done' signal")
            progress_queue.put(("done", None))

    thread = threading.Thread(target=run_retrieval, daemon=True)
    thread.start()
    print("[ASK] Retrieval thread launched, waiting for progress…")

    # Stream progress updates while retrieval runs
    while True:
        msg_type, payload = progress_queue.get()
        if msg_type == "progress":
            print(f"[QUEUE] Progress update received, yielding to Gradio")
            history[-1]["content"] = payload
            yield history, "", "", ""
        elif msg_type == "done":
            print("[QUEUE] 'done' signal received, exiting progress loop")
            break

    thread.join()
    print("[ASK] Thread joined")

    if not result_holder:
        print("[ASK] ERROR: result_holder is empty!")
        history[-1]["content"] = "❌ Retrieval returned no result."
        yield history, "", "", ""
        return

    if "error" in result_holder[0]:
        print(f"[ASK] ERROR from retrieval: {result_holder[0]['error']}")
        history[-1]["content"] = f"❌ Retrieval error: {result_holder[0]['error']}"
        yield history, "", "", ""
        return

    result = result_holder[0]
    print(f"[ASK] rewritten_query: {result['rewritten_query']!r}")
    print(f"[ASK] hyde_passage (first 100 chars): {result['hyde_passage'][:100]!r}")
    print(f"[ASK] {len(result['results'])} reranked results")
    yield history, result["rewritten_query"], result["hyde_passage"], ""

    # Build context from top-k results
    context_blocks = "\n\n".join(
        f"**[{i+1}]** (score {r.get('rerank_score', 0):.3f}, "
        f"file: {r.get('filename', '?')})\n{r.get('chunk_text', '')}"
        for i, r in enumerate(result["results"])
    )
    print(f"[ASK] Context built ({len(context_blocks)} chars). Calling LLM…")

    system_msg = (
        "You are a precise, helpful assistant. "
        "Answer the question using ONLY the provided context. "
        "Cite the numbered passages when relevant."
    )
    user_msg = f"Context:\n{context_blocks}\n\nQuestion: {query}"

    pipeline = get_retrieval_pipeline()
    print(f"[LLM] Streaming answer with model: {cfg.HEAVY_WEIGHT_MODEL}")
    stream = pipeline.nvidia_client.chat.completions.create(
        model=cfg.HEAVY_WEIGHT_MODEL,
        messages=[
            {"role": "system",  "content": system_msg},
            {"role": "user",    "content": user_msg},
        ],
        stream=True,
        max_tokens=1024,
        temperature=0.2,
    )

    answer = ""
    token_count = 0
    for chunk in stream:
        delta  = chunk.choices[0].delta.content or ""
        answer += delta
        token_count += 1
        if token_count % 20 == 0:
            print(f"[LLM] ...streamed {token_count} tokens so far")
        history[-1]["content"] = answer
        yield history, result["rewritten_query"], result["hyde_passage"], context_blocks
    print(f"[LLM] Stream complete. Total tokens: {token_count}")



# ══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
body { font-family: 'Inter', sans-serif; }
.tab-nav button { font-weight: 600; font-size: 1rem; }
.stage-log { font-family: monospace; font-size: 0.875rem; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="RAG Pipeline",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=CSS,
) as demo:

    gr.Markdown(
        "# 🔮 RAG Pipeline\n"
        "Upload documents → automatic ingestion → semantically search and answer."
    )

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with gr.Tab("📤  Upload & Ingest"):
        gr.Markdown(
            "Upload PDFs (or other supported formats). Files are stored idempotently "
            "and processed by a background Celery worker. Watch the live stage log below."
        )
        file_input = gr.Files(
            label="Drop files here",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt"],
        )
        ingest_btn = gr.Button("🚀  Start Ingestion", variant="primary", size="lg")
        ingest_log = gr.Markdown(
            label="Stage log",
            value="*Waiting for files…*",
            elem_classes=["stage-log"],
        )

        ingest_btn.click(
            fn=upload_and_ingest,
            inputs=[file_input],
            outputs=[ingest_log],
        )

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with gr.Tab("💬  Ask"):
        gr.Markdown(
            "Ask a question. The pipeline rewrites your query, generates a HyDE passage, "
            "retrieves top-30 candidates, reranks to top-5, then streams the answer."
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            type="messages",
            height=420,
            show_copy_button=True,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=rag"),
        )

        with gr.Row():
            query_box = gr.Textbox(
                placeholder="Ask a question about your documents…",
                label="Question",
                scale=5,
                lines=1,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)

        with gr.Accordion("🔬 Retrieval debug", open=False):
            rewritten_box  = gr.Textbox(label="Rewritten query",  lines=2, interactive=False)
            hyde_box       = gr.Textbox(label="HyDE passage",      lines=5, interactive=False)
            context_box    = gr.Textbox(label="Top-5 context chunks", lines=12, interactive=False)

        ask_btn.click(
            fn=ask_stream,
            inputs=[query_box, chatbot],
            outputs=[chatbot, rewritten_box, hyde_box, context_box],
        )
        query_box.submit(
            fn=ask_stream,
            inputs=[query_box, chatbot],
            outputs=[chatbot, rewritten_box, hyde_box, context_box],
        )


if __name__ == "__main__":
    demo.queue().launch(share=False, server_port=7860)
