"""
Ingestion pipeline (run this once per new batch of documents).

Stages:
  1. Load + semantically chunk documents
  2. Extract entities/relationships via a TRUE Gemini Batch API job
     (the ONLY step that uses Gemini in this whole project)
  3. Store the extracted graph in Neo4j
  4. Run Leiden community detection (Neo4j GDS)
  5. Summarize each community with the LOCAL LLM
  6. Store community summaries as :Community nodes in Neo4j
  7. Build chunk + community vector indexes directly in Neo4j (HNSW ANN index)

Usage:
    python ingest.py
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_neo4j import Neo4jVector
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_unstructured.document_loaders import UnstructuredLoader

from config import (
    BASE_DIR,
    BATCH_DIR,
    BATCH_POLL_SECONDS,
    COMMUNITY_PROPERTY,
    FORBIDDEN_CHUNK_CATEGORIES,
    GDS_GRAPH_NAME,
    GEMINI_EXTRACTION_MODEL,
    GOOGLE_API_KEY,
    LOCAL_MODEL,
    MAX_CHARS_PER_CHUNK,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    UNSTRUCTURED_API_KEY,
    embeddings,
    graph,
    local_llm,
)

genai_client = genai.Client(api_key=GOOGLE_API_KEY)  # Gemini used ONLY here, for the batch job
splitter = SemanticChunker(embeddings=embeddings)


# --------------------------------------------------------------------------
# Stage 1: Load + chunk documents
# --------------------------------------------------------------------------

def load_files(directory: str) -> list[Document]:
    if not Path(directory).exists():
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    loader_kwargs: dict[str, Any] = {"strategy": "auto"}
    if UNSTRUCTURED_API_KEY:
        loader_kwargs.update({"api_key": UNSTRUCTURED_API_KEY, "partition_via_api": True})

    loader = DirectoryLoader(
        directory,
        glob="**/*",
        recursive=True,
        loader_cls=UnstructuredLoader,
        loader_kwargs=loader_kwargs,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} raw documents.")

    filtered: list[Document] = []
    for doc in tqdm(docs, desc="Filtering documents"):
        category = doc.metadata.get("category")
        content = (doc.page_content or "").strip()
        if category in FORBIDDEN_CHUNK_CATEGORIES or len(content) <= 1 or content.isdigit():
            continue
        doc.metadata["source_path"] = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
        filtered.append(doc)

    print(f"Kept {len(filtered)} cleaned documents.")
    return filtered


def semantic_chunking(docs: list[Document]) -> list[Document]:
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata)
        chunk.metadata["chunk_id"] = f"chunk-{index:06d}"
    print(f"Created {len(chunks)} semantic chunks.")
    return chunks


# --------------------------------------------------------------------------
# Stage 2: TRUE batch extraction via Gemini Batch API
# --------------------------------------------------------------------------

class ExtractedNode(BaseModel):
    id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelationship(BaseModel):
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ChunkExtraction(BaseModel):
    chunk_id: str
    nodes: list[ExtractedNode] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = (
    "You extract knowledge-graph data from a single document chunk. "
    "Return only facts explicitly supported by the text. "
    "Use short relationship types such as MENTIONS, WORKS_FOR, LOCATED_IN, CAUSES, USES. "
    "If nothing graph-worthy is present, return empty nodes and relationships."
)


def build_batch_request_file(chunks: list[Document]) -> Path:
    """One JSONL line per chunk = one request in the batch job."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    request_path = BATCH_DIR / f"extraction_requests_{int(time.time())}.jsonl"

    with open(request_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Preparing JSONL file for batch job"):
            chunk_id = chunk.metadata["chunk_id"]
            text = chunk.page_content[:MAX_CHARS_PER_CHUNK]
            request_line = {
                "key": chunk_id,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{EXTRACTION_SYSTEM_PROMPT}\n\nChunk text:\n{text}"}],
                        }
                    ],
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_schema": ChunkExtraction.model_json_schema(),
                    },
                },
            }
            f.write(json.dumps(request_line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} batch requests to {request_path}")
    return request_path


def submit_and_wait_for_batch(request_path: Path) -> str:
    uploaded_file = genai_client.files.upload(
        file=str(request_path),
        config={"display_name": request_path.stem, "mime_type": "jsonl"},
    )
    batch_job = genai_client.batches.create(
        model=GEMINI_EXTRACTION_MODEL,
        src=uploaded_file.name,
        config={"display_name": f"extraction-{request_path.stem}"},
    )
    print(f"Submitted batch job: {batch_job.name}")

    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    while True:
        batch_job = genai_client.batches.get(name=batch_job.name)
        state = batch_job.state.name
        print(f"  batch state: {state}")
        if state in terminal_states:
            break
        time.sleep(BATCH_POLL_SECONDS)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job ended in state {batch_job.state.name}")

    return batch_job.dest.file_name


def parse_batch_results(result_file_name: str) -> dict[str, ChunkExtraction]:
    raw_bytes = genai_client.files.download(file=result_file_name)
    lines = raw_bytes.decode("utf-8").splitlines()

    extractions: dict[str, ChunkExtraction] = {}
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        response = row.get("response")
        if not key or not response:
            print(f"Skipping failed batch row: {row.get('error')}")
            continue
        candidate_text = response["candidates"][0]["content"]["parts"][0]["text"]
        try:
            parsed = ChunkExtraction.model_validate_json(candidate_text)
        except Exception as exc:
            print(f"Failed to parse extraction for {key}: {exc}")
            continue
        extractions[key] = parsed

    print(f"Parsed {len(extractions)} chunk extractions from batch results.")
    return extractions


def chunk_to_graph_document(chunk_doc: Document, extraction: ChunkExtraction) -> GraphDocument:
    nodes = [Node(id=n.id, type=n.type, properties=n.properties or {}) for n in extraction.nodes]
    lookup = {(n.id, n.type): n for n in nodes}

    relationships = []
    for rel in extraction.relationships:
        source = lookup.get((rel.source_id, rel.source_type)) or Node(id=rel.source_id, type=rel.source_type)
        target = lookup.get((rel.target_id, rel.target_type)) or Node(id=rel.target_id, type=rel.target_type)
        relationships.append(Relationship(source=source, target=target, type=rel.type, properties=rel.properties or {}))

    return GraphDocument(nodes=nodes, relationships=relationships, source=chunk_doc)


def extract_graph_documents_via_batch(chunks: list[Document]) -> list[GraphDocument]:
    """Runs the actual Gemini Batch API job (async, ~50% cheaper) instead of per-chunk calls."""
    request_path = build_batch_request_file(chunks)
    result_file_name = submit_and_wait_for_batch(request_path)
    extractions = parse_batch_results(result_file_name)

    graph_documents = []
    for chunk in chunks:
        extraction = extractions.get(chunk.metadata["chunk_id"])
        if extraction is None:
            continue
        graph_documents.append(chunk_to_graph_document(chunk, extraction))
    return graph_documents


# --------------------------------------------------------------------------
# Stage 4: Leiden community detection (Neo4j GDS)
# --------------------------------------------------------------------------

def drop_projection(graph_name: str) -> None:
    try:
        graph.query("CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName", {"graph_name": graph_name})
    except Exception:
        pass


def run_leiden() -> None:
    drop_projection(GDS_GRAPH_NAME)
    projection_stats = graph.query(
        """
        CALL gds.graph.project.cypher(
          $graph_name,
          'MATCH (n:__Entity__) RETURN id(n) AS id, labels(n) AS labels',
          'MATCH (s:__Entity__)-[r]->(t:__Entity__) RETURN id(s) AS source, id(t) AS target, type(r) AS type'
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {"graph_name": GDS_GRAPH_NAME},
    )
    print("Projection:", projection_stats)

    leiden_stats = graph.query(
        """
        CALL gds.leiden.write($graph_name, {writeProperty: $community_property})
        YIELD communityCount, modularity
        RETURN communityCount, modularity
        """,
        {"graph_name": GDS_GRAPH_NAME, "community_property": COMMUNITY_PROPERTY},
    )
    print("Leiden result:", leiden_stats)


def fetch_community_rows() -> list[dict[str, Any]]:
    return graph.query(
        f"""
        MATCH (entity:__Entity__)
        WHERE entity.{COMMUNITY_PROPERTY} IS NOT NULL
        OPTIONAL MATCH (entity)-[r]->(neighbor:__Entity__)
        WHERE neighbor.{COMMUNITY_PROPERTY} = entity.{COMMUNITY_PROPERTY}
        WITH entity.{COMMUNITY_PROPERTY} AS community_id,
             collect(DISTINCT entity.id) AS members,
             collect(DISTINCT CASE WHEN r IS NULL THEN NULL ELSE {{source: entity.id, type: type(r), target: neighbor.id}} END) AS relationships
        RETURN community_id, members, [item IN relationships WHERE item IS NOT NULL][0..100] AS relationships
        ORDER BY community_id
        """
    )


# --------------------------------------------------------------------------
# Stage 5-6: Summarize communities with the LOCAL LLM, store as :Community nodes
# --------------------------------------------------------------------------

COMMUNITY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You summarize graph communities for retrieval. Name the theme, key entities, and main relationships concisely."),
        ("human", "Community id: {community_id}\n\nMembers:\n{members}\n\nRelationships:\n{relationships}\n\nWrite the summary."),
    ]
)
community_summary_chain = COMMUNITY_SUMMARY_PROMPT | local_llm | StrOutputParser()


def summarize_and_store_communities(community_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for row in tqdm(community_rows, desc="Summarizing communities (local LLM)"):
        summary = community_summary_chain.invoke(
            {
                "community_id": row["community_id"],
                "members": json.dumps(row["members"], ensure_ascii=False, indent=2),
                "relationships": json.dumps(row["relationships"], ensure_ascii=False, indent=2),
            }
        )
        summaries.append({**row, "summary": summary})

    if summaries:
        graph.query(
            """
            UNWIND $rows AS row
            MERGE (c:Community {community_id: row.community_id})
            SET c.summary = row.summary, c.member_count = size(row.members),
                c.model = $model, c.updated_at = datetime()
            WITH c, row
            UNWIND row.members AS member_id
            MATCH (e:__Entity__ {id: member_id})
            MERGE (e)-[:IN_COMMUNITY]->(c)
            """,
            {"rows": summaries, "model": LOCAL_MODEL},
        )
        print(f"Stored {len(summaries)} community summaries in Neo4j.")
    return summaries


# --------------------------------------------------------------------------
# Stage 7: Vector indexes — stored directly in Neo4j (HNSW ANN index),
# attached to the SAME nodes already created above (no separate vector DB).
# --------------------------------------------------------------------------

def new_index_name(prefix: str) -> str:
    """Generates a fresh, collision-safe index name for this ingestion run."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def build_chunk_vector_index(index_name: str) -> Neo4jVector:
    return Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )


def build_community_vector_index(index_name: str, community_summaries: list[dict[str, Any]]) -> Neo4jVector | None:
    if not community_summaries:
        return None
    return Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        node_label="Community",
        text_node_properties=["summary"],
        embedding_node_property="embedding",
    )


# --------------------------------------------------------------------------
# Reusable pipeline entrypoint (used by both CLI `main()` and the Gradio app)
# --------------------------------------------------------------------------

def run_ingestion_pipeline(directory: str, progress_callback=None) -> dict[str, Any]:
    """
    Runs the full ingestion pipeline on `directory` and returns a dict with:
      chunk_index_name, community_index_name, chunk_store, community_store, log
    A fresh random index name is generated every call, so re-running never
    collides with a previous run's vector index.
    """
    log: list[str] = []

    def emit(message: str) -> None:
        log.append(message)
        print(message)
        if progress_callback is not None:
            progress_callback(message)

    emit(f"Loading documents from {directory} ...")
    source_documents = load_files(directory)
    chunks = semantic_chunking(source_documents)
    emit(f"Created {len(chunks)} semantic chunks.")

    emit("Submitting Gemini batch extraction job ...")
    graph_documents = extract_graph_documents_via_batch(chunks)
    emit(f"Extracted {len(graph_documents)} graph documents from {len(chunks)} chunks.")

    graph.query("RETURN 1 AS ok")
    graph.add_graph_documents(graph_documents=graph_documents, include_source=True, baseEntityLabel=True)
    emit("Stored graph documents in Neo4j.")

    emit("Running Leiden community detection ...")
    run_leiden()
    community_rows = fetch_community_rows()
    emit(f"Detected {len(community_rows)} communities.")

    community_summaries = summarize_and_store_communities(community_rows)
    emit(f"Summarized {len(community_summaries)} communities with the local LLM.")

    chunk_index_name = new_index_name("chunk_vector_index")
    community_index_name = new_index_name("community_vector_index") if community_summaries else None

    chunk_store = build_chunk_vector_index(chunk_index_name)
    community_store = (
        build_community_vector_index(community_index_name, community_summaries)
        if community_index_name
        else None
    )
    emit(f"Built chunk vector index: {chunk_index_name}")
    if community_index_name:
        emit(f"Built community vector index: {community_index_name}")
    else:
        emit("No communities detected — skipped community vector index.")

    return {
        "chunk_index_name": chunk_index_name,
        "community_index_name": community_index_name,
        "chunk_store": chunk_store,
        "community_store": community_store,
        "log": log,
    }


# --------------------------------------------------------------------------
# Main (CLI usage)
# --------------------------------------------------------------------------

def main() -> None:
    result = run_ingestion_pipeline(str(BASE_DIR / "documents"))
    print("\nIngestion complete.")
    print(f"Chunk vector index:     {result['chunk_index_name']}")
    print(f"Community vector index: {result['community_index_name']}")
    print("\nStart the UI with `python app.py`, then paste these index names in "
          "if prompted, or drive ingestion directly from the app's upload box.")


if __name__ == "__main__":
    main()