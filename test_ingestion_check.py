"""
Test script for ingestion pipeline chunk inspection.

1. Inspects raw JSON node structure returned by PDFLoader after PDF parsing.
2. Inspects final Document chunks produced by SemanticChunker.
3. Traces image and table nodes to explain where they are located.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from config import MAX_CHARS, OVERLAP_CHARS, UPLOAD_DIR
from ingestion.chunking import SemanticChunker
from ingestion.loader import PDFLoader


def traverse_raw_ast(node: dict, type_counts: Counter, samples: dict, path: str = "root"):
    """Recursively inspect raw AST nodes produced by OpenDataLoader PDF parser."""
    node_type = node.get("type", "unknown")
    type_counts[node_type] += 1

    if node_type in ("image", "figure", "picture", "table", "table_block"):
        if node_type not in samples:
            samples[node_type] = []
        if len(samples[node_type]) < 3:
            samples[node_type].append((node, path))

    for idx, child in enumerate(node.get("kids", [])):
        traverse_raw_ast(child, type_counts, samples, path=f"{path}.kids[{idx}]")



def test_ingestion_check(target_path: str):
    print("=" * 70)
    print(f"Testing ingestion pipeline for target: {target_path}")
    print("=" * 70)

    path = Path(target_path)
    if not path.exists():
        print(f"Error: Target path '{target_path}' does not exist.")
        return

    # ── STEP 1: PARSING (PDFLoader) ──────────────────────────────────────────
    print("\n[STEP 1] Loading document(s) with PDFLoader...")
    loader = PDFLoader()
    docs = loader.load(str(path))
    print(f"   Loaded {len(docs)} document(s).")

    raw_node_counts = Counter()
    raw_node_samples = {}

    for i, doc in enumerate(docs, 1):
        try:
            raw_data = json.loads(doc.page_content)
            file_name = raw_data.get("file name", f"Doc {i}")
            traverse_raw_ast(raw_data, raw_node_counts, raw_node_samples)
        except Exception as e:
            print(f"   Doc {i} raw content is not JSON or failed to parse: {e}")

    print("\n" + "=" * 70)
    print("RAW AST NODE TYPES (Found right after PDFLoader parsing):")
    print("=" * 70)
    for ntype, count in raw_node_counts.most_common():
        print(f"  • {ntype:20s}: {count} node(s)")

    print("\n" + "=" * 70)
    print("SAMPLES OF RAW IMAGE / TABLE NODES IN PARSED JSON:")
    print("=" * 70)
    if not raw_node_samples:
        print("  No raw image/table nodes found in the parsed JSON AST.")
    else:
        for ntype, sample_list in raw_node_samples.items():
            print(f"\n--- Node Type: '{ntype.upper()}' ({len(sample_list)} sample(s)) ---")
            for idx, (node, npath) in enumerate(sample_list, 1):
                keys = list(node.keys())
                num_kids = len(node.get("kids", []))
                print(f" Sample #{idx} at AST path: {npath}")
                print(f"   Keys available : {keys}")
                print(f"   Child Kids count: {num_kids}")

                # For table nodes: dump the full structure to understand rows format
                if ntype == "table":
                    rows_val = node.get("rows")
                    print(f"   'rows' key type: {type(rows_val).__name__}")
                    if isinstance(rows_val, list) and rows_val:
                        print(f"   rows count      : {len(rows_val)}")
                        first_row = rows_val[0]
                        print(f"   First row type  : {type(first_row).__name__}")
                        if isinstance(first_row, dict):
                            print(f"   First row keys  : {list(first_row.keys())}")
                            # Look inside first row's children
                            for sub_key in ("kids", "cells", "columns"):
                                sub = first_row.get(sub_key)
                                if sub:
                                    print(f"   First row['{sub_key}'][0] keys: {list(sub[0].keys()) if isinstance(sub[0], dict) else sub[0]}")
                                    break
                        elif isinstance(first_row, list):
                            print(f"   First row (list) sample: {json.dumps(first_row[:2], default=str)[:200]}")
                    elif isinstance(rows_val, dict):
                        print(f"   rows keys       : {list(rows_val.keys())}")
                    print(f"   Full raw dump   : {json.dumps(node, default=str)[:500]}")


    # ── STEP 2: CHUNKING (SemanticChunker) ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("[STEP 2] Chunking document(s) with SemanticChunker...")
    print("=" * 70)
    chunker = SemanticChunker(max_chars=MAX_CHARS, overlap_chars=OVERLAP_CHARS)

    all_chunks = []
    for i, doc in enumerate(docs, 1):
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        print(f"   Doc {i}: generated {len(chunks)} chunk(s).")

    print(f"\nTotal chunks generated across all documents: {len(all_chunks)}")

    # ── STEP 3: CHUNK SUMMARY ─────────────────────────────────────────────────
    chunk_type_counts = Counter()
    special_chunks = []

    for idx, chunk in enumerate(all_chunks):
        chunk_type = chunk.metadata.get("type", "unknown")
        chunk_type_counts[chunk_type] += 1
        if chunk_type in ("image", "table", "figure", "picture"):
            special_chunks.append((idx, chunk))

    print("\n" + "=" * 70)
    print("FINAL CHUNK TYPE SUMMARY (Produced by SemanticChunker):")
    print("=" * 70)
    for ctype, count in chunk_type_counts.most_common():
        print(f"  • {ctype:20s}: {count} chunk(s)")

    print("\n" + "=" * 70)
    print(f"FINAL SPECIAL CHUNKS FOUND (image / table / figure): {len(special_chunks)}")
    print("=" * 70)

    if not special_chunks:
        print("  No image or table chunks were generated.")
    else:
        for idx, chunk in special_chunks:
            meta = chunk.metadata
            ctype = meta.get("type")
            page = meta.get("page", "?")
            fname = meta.get("file_name", "?")
            bbox = meta.get("bbox", None)
            img_path = meta.get("image_path", None)
            snippet = chunk.page_content[:150].replace("\n", " ")

            print(f"\n[Chunk #{idx}] Type: '{ctype.upper()}' | Page: {page} | File: {fname}")
            if bbox:
                print(f"   BBox      : {bbox}")
            if img_path:
                print(f"   Image Path: {img_path}")
            print(f"   Content   : {snippet!r}...")

    print("\n" + "=" * 70)
    print("PIPELINE STOPPED (Skipped Weaviate Vector Upsert).")
    print("=" * 70)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else UPLOAD_DIR
    test_ingestion_check(target)

