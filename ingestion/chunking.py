"""
Document chunking for the multimodal RAG pipeline.

Processes a :class:`~ingestion.models.ParsedDocument` into LangChain
``Document`` chunks:

1. **Text chunks**    — paragraph, heading, list (one chunk each).
2. **Table chunks**   — each table rendered as Markdown, kept whole.
3. **Image chunks**   — one chunk per image (empty ``page_content``).
4. **Caption chunks** — caption text cross-referenced to its image via
                        ``linked_image_id`` in metadata.

Metadata on every chunk is minimal:
  ``file_name``, ``bbox``, ``id``, ``type``
  + ``image_path``      (image only)
  + ``linked_image_id`` (caption only)
"""

import logging
from typing import Any, Generator

from langchain_core.documents import Document

from ingestion.models import ParsedDocument

logger = logging.getLogger("ingestion.chunking")

_TEXT_TYPES = {"paragraph", "heading", "caption", "list_item"}


def walk(kids: list[dict[str, Any]], etypes: str | set[str] | tuple[str, ...]) -> Generator[dict, None, None]:
    """Depth-first traversal, yielding elements whose ``type`` matches *etypes*."""
    if isinstance(etypes, str):
        target_types = {etypes}
    else:
        target_types = set(etypes)

    for elem in kids:
        if elem.get("type") in target_types:
            yield elem
        for child_key in ("kids", "list_items"):
            if child_key in elem:
                yield from walk(elem[child_key], target_types)


def table_to_markdown(table: dict[str, Any]) -> str:
    rows: list[dict] = table.get("rows", [])
    if not rows:
        return ""

    md_rows: list[list[str]] = []
    for row in rows:
        cells = row.get("cells", [])
        cell_texts: list[str] = []
        for cell in cells:
            parts = [
                kid.get("content", "").strip()
                for kid in cell.get("kids", [])
                if kid.get("type") in _TEXT_TYPES
            ]
            cell_texts.append(" ".join(parts))
        md_rows.append(cell_texts)

    if not md_rows:
        return ""

    header = md_rows[0]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in md_rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def list_to_text(list_elem: dict[str, Any]) -> str:
    items = list_elem.get("list_items", [])
    parts = [item.get("content", "").strip() for item in items if item.get("content")]
    return "\n\n".join(parts)


class DocumentChunker:
    def __init__(self):
        pass

    def text_chunks(self, doc: ParsedDocument) -> list[Document]:
        chunks: list[Document] = []
        print("Document: ", doc)
        for elem in walk(doc.kids, {"paragraph", "heading", "list"}):
            etype = elem.get("type", "")
            if etype in ("paragraph", "heading"):
                text = elem.get("content", "").strip()
            elif etype == "list":
                text = list_to_text(elem)
            else:
                continue
            if not text:
                continue
            chunks.append(Document(
                page_content=text,
                metadata={
                    "file_name": doc.file_name,
                    "bbox": elem.get("bounding box"),
                    "id": elem.get("id"),
                    "type": etype,
                },
            ))
        return chunks

    def table_chunks(self, doc: ParsedDocument) -> list[Document]:
        chunks: list[Document] = []
        for table in walk(doc.kids, "table"):
            md = table_to_markdown(table)
            if not md.strip():
                continue
            chunks.append(Document(
                page_content=md,
                metadata={
                    "file_name": doc.file_name,
                    "bbox": table.get("bounding box"),
                    "id": table.get("id"),
                    "type": "table",
                },
            ))
        return chunks

    def image_chunks(self, doc: ParsedDocument) -> list[Document]:
        chunks: list[Document] = []
        for img in walk(doc.kids, "image"):
            path = img.get("source") or img.get("data", "")
            if not path:
                continue
            chunks.append(Document(
                page_content="",
                metadata={
                    "file_name": doc.file_name,
                    "bbox": img.get("bounding box"),
                    "id": img.get("id"),
                    "type": "image",
                    "image_path": path,
                },
            ))
        return chunks

    def caption_chunks(
        self,
        doc: ParsedDocument,
        image_chunks: list[Document],
    ) -> list[Document]:
        img_id_map: dict[int, Any] = {
            img.get("id"): chunk.metadata["id"]
            for img, chunk in zip(walk(doc.kids, "image"), image_chunks)
            if img.get("id") is not None
        }

        chunks: list[Document] = []
        for cap in walk(doc.kids, "caption"):
            text = cap.get("content", "").strip()
            if not text:
                continue
            linked_elem_id  = cap.get("linked_content_id")
            linked_image_id = img_id_map.get(linked_elem_id) if linked_elem_id else None
            chunks.append(Document(
                page_content=text,
                metadata={
                    "file_name": doc.file_name,
                    "bbox": cap.get("bounding box"),
                    "id": cap.get("id"),
                    "type": "caption",
                    "linked_image_id": linked_image_id,
                },
            ))
        return chunks

    # ── public API ─────────────────────────────────────────────────────────────

    def process_document(self, doc: ParsedDocument) -> list[Document]:
        text_chunks    = self.text_chunks(doc)
        table_chunks   = self.table_chunks(doc)
        image_chunks   = self.image_chunks(doc)
        caption_chunks = self.caption_chunks(doc, image_chunks)

        all_chunks = text_chunks + table_chunks + image_chunks + caption_chunks

        logger.info(
            "process_document(%s): text=%d  tables=%d  images=%d  captions=%d",
            doc.file_name,
            len(text_chunks),
            len(table_chunks),
            len(image_chunks),
            len(caption_chunks),
        )

        return all_chunks