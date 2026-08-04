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
from openai import AsyncOpenAI, OpenAI


import logging

from config import *
from storage.object_storage import ObjectStorage
from storage.weaviate import WeaviateVDB
from ingestion.embedding import MultimodalEncoder
from retrieval.pipeline import RetrievalPipeline
from tasks import celery_app, run_ingestion as celery_run_ingestion  # noqa: F401

from agent.workflow import CoodingAgent
from langchain_core.messages import HumanMessage

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

storage = ObjectStorage(s3_uri=S3_URI, local_dir=UPLOAD_DIR)

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
        
        encoder = MultimodalEncoder(device=DEVICE, batch_size=EMBED_BATCH, model_name=EMBEDDING_MODEL)
        vdb = WeaviateVDB(WEAVIATE_REST_ENDPOINT, WEAVIATE_API_KEY, INDEX_NAME, encoder.d_model)
        
        qwen_base_url = QWEN_SERVER_URL.rstrip("/")
        if not qwen_base_url.endswith("/v1"):
            qwen_base_url += "/v1"
            
        client = AsyncOpenAI(api_key="EMPTY", base_url=qwen_base_url)
        
        _retrieval_pipeline = RetrievalPipeline(
            encoder=encoder,
            vdb=vdb,
            local_client=client,
            local_model=QWEN_MODEL,
            retrieve_top_k=RETRIEVE_TOP_K,
            rerank_top_k=RERANK_TOP_K,
            reranker=pre_loaded_reranker,
        )
    return _retrieval_pipeline


# ── Stage decorations ──────────────────────────────────────────────────────────
ICONS = {
    "DOWNLOADING": "📥",
    "PARSING":     "📄",
    "STT":         "🎙️",
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


def _format_source(result: dict) -> str:
    filename = result.get("filename", "?")
    timestamp = result.get("video_timestamp")
    start_minute = result.get("video_start_minute")
    if timestamp:
        minute_label = f"minute {start_minute}" if start_minute is not None else "video"
        return f"video: {filename} @ {timestamp} ({minute_label})"
    pages = result.get("page_numbers")
    if pages:
        return f"file: {filename}, pages: {pages}"
    return f"file: {filename}"


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

coding_agent = CoodingAgent()

async def ensure_agent_ready():
    if coding_agent.coding_agent is None:
        await coding_agent.init_agent()

async def agent_stream(query: str, history: list):
    if not query.strip():
        yield history, "", "", ""
        return

    print(f"\n{'='*60}")
    print(f"[ASK] Query received: {query!r}")
    print(f"{'='*60}")

    history = list(history or [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": "⏳ Thinking..."})
    yield history, "", "", ""

    await ensure_agent_ready()

    try:
        async for chunk in coding_agent.arun([HumanMessage(content=query)]):
            if history[-1]["content"] == "⏳ Thinking...":
                history[-1]["content"] = chunk
            else:
                history[-1]["content"] += chunk
            yield history, "", "", ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        history[-1]["content"] += f"\n\n❌ Error: {str(e)}"
        yield history, "", "", ""


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
            file_types=[
                ".pdf", ".docx", ".txt",
                ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
                ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
            ],
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
            fn=agent_stream,
            inputs=[query_box, chatbot],
            outputs=[chatbot, rewritten_box, hyde_box, context_box],
        )
        query_box.submit(
            fn=agent_stream,
            inputs=[query_box, chatbot],
            outputs=[chatbot, rewritten_box, hyde_box, context_box],
        )


if __name__ == "__main__":
    demo.queue().launch(share=False, server_port=7860)
