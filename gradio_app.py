"""
Gradio app: upload documents -> run ingestion pipeline (random index names
each run) -> ask questions, answered entirely by the local model.

Gemini is used only inside ingestion_pipeline.py, only for the batch extraction step.
Everything in this file uses the local model.

Usage:
    python app.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jVector

from config import graph, local_llm
import ingestion_pipeline

# --------------------------------------------------------------------------
# Session state — populated once ingestion has been run from the UI.
# Index names are random per run, so we keep the live store objects here
# rather than assuming a fixed name exists in Neo4j.
# --------------------------------------------------------------------------

state = {
    "chunk_store": None,
    "community_store": None,
    "chunk_index_name": None,
    "community_index_name": None,
}


def run_ingestion_ui(files: list[str], progress=gr.Progress(track_tqdm=False)):
    if not files:
        return "Please upload at least one file first.", gr.update(interactive=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for file_path in files:
            shutil.copy(file_path, tmp_path / Path(file_path).name)

        log_lines: list[str] = []

        def on_progress(message: str) -> None:
            log_lines.append(message)
            progress(0, desc=message)

        result = ingestion_pipeline.run_ingestion_pipeline(str(tmp_path), progress_callback=on_progress)

    state["chunk_store"] = result["chunk_store"]
    state["community_store"] = result["community_store"]
    state["chunk_index_name"] = result["chunk_index_name"]
    state["community_index_name"] = result["community_index_name"]

    summary = (
        f"Ingestion complete.\n"
        f"Chunk vector index: {result['chunk_index_name']}\n"
        f"Community vector index: {result['community_index_name'] or '(none — no communities detected)'}\n\n"
        + "\n".join(result["log"])
    )
    return summary, gr.update(interactive=True)


# --------------------------------------------------------------------------
# Local search (vector search over chunks -> traverse mentioned entities)
# --------------------------------------------------------------------------

def local_search(question: str, top_k: int = 5, depth: int = 2) -> str:
    chunk_store: Neo4jVector = state["chunk_store"]
    similar_chunks = chunk_store.similarity_search(question, k=top_k)
    context_blocks = []
    for chunk in similar_chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            context_blocks.append(chunk.page_content)
            continue
        rows = graph.query(
            f"""
            MATCH (d:Document {{chunk_id: $chunk_id}})-[:MENTIONS]->(e)
            OPTIONAL MATCH (e)-[r*1..{depth}]-(related)
            RETURN d.text AS chunk_text, e.id AS entity,
                   [x IN r | type(x)] AS relationship_path,
                   related.id AS related_entity
            LIMIT 25
            """,
            {"chunk_id": chunk_id},
        )
        context_blocks.append(json.dumps(rows, ensure_ascii=False))
    return "\n\n".join(context_blocks)


def global_search(question: str, top_k: int = 3) -> str:
    community_store: Neo4jVector | None = state["community_store"]
    if community_store is None:
        return ""
    similar_communities = community_store.similarity_search(question, k=top_k)
    return "\n\n".join(doc.page_content for doc in similar_communities)


# --------------------------------------------------------------------------
# Answer chain — LOCAL model only
# --------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the provided context. Cite specific entities/relationships when relevant."),
        ("human", "Local (entity-level) context:\n{local_context}\n\nGlobal (community-level) context:\n{global_context}\n\nQuestion: {question}"),
    ]
)
answer_chain = ANSWER_PROMPT | local_llm | StrOutputParser()


def graph_rag_answer(question: str) -> tuple[str, str, str]:
    local_context = local_search(question)
    global_context = global_search(question)
    answer = answer_chain.invoke(
        {"local_context": local_context, "global_context": global_context, "question": question}
    )
    return answer, local_context, global_context


def respond(question: str, show_context: bool):
    if state["chunk_store"] is None:
        return "Upload documents and run ingestion first (above).", "", ""
    if not question.strip():
        return "Please enter a question.", "", ""
    answer, local_context, global_context = graph_rag_answer(question)
    if not show_context:
        return answer, "", ""
    return answer, local_context, global_context


# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------

with gr.Blocks(title="GraphRAG Explorer") as demo:
    gr.Markdown(
        "# GraphRAG Explorer\n"
        "1. Upload documents and run ingestion (uses Gemini once, for batch extraction).\n"
        "2. Ask questions — answered entirely by the local model.\n"
        "Each ingestion run builds a fresh, uniquely-named vector index."
    )

    with gr.Accordion("Step 1: Ingest documents", open=True):
        file_upload = gr.Files(label="Upload documents", file_count="multiple")
        ingest_button = gr.Button("Run ingestion", variant="primary")
        ingest_log = gr.Textbox(label="Ingestion log", lines=12)

    gr.Markdown("---")

    with gr.Accordion("Step 2: Ask questions", open=True):
        with gr.Row():
            question_box = gr.Textbox(label="Question", placeholder="e.g. What are the main themes discussed?", scale=4)
            ask_button = gr.Button("Ask", variant="primary", scale=1, interactive=False)

        show_context_checkbox = gr.Checkbox(label="Show retrieved context (local + global)", value=False)
        answer_box = gr.Textbox(label="Answer", lines=8)

        with gr.Accordion("Retrieved context (debug)", open=False):
            local_context_box = gr.Textbox(label="Local search context (entities/relationships)", lines=10)
            global_context_box = gr.Textbox(label="Global search context (community summaries)", lines=10)

    ingest_button.click(
        fn=run_ingestion_ui,
        inputs=[file_upload],
        outputs=[ingest_log, ask_button],
    )

    ask_button.click(
        fn=respond,
        inputs=[question_box, show_context_checkbox],
        outputs=[answer_box, local_context_box, global_context_box],
    )
    question_box.submit(
        fn=respond,
        inputs=[question_box, show_context_checkbox],
        outputs=[answer_box, local_context_box, global_context_box],
    )


if __name__ == "__main__":
    demo.launch()