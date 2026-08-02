"""
PDF loading via OpenDataLoaderPDFLoader (LangChain integration).

Wraps the loader with the exact configuration for the multimodal RAG
pipeline:
  - format="json"          → element-level structure (type, bbox, page)
  - image_output="external" → images saved to disk
  - image_format="png"
  - table_method="cluster"  → AI-assisted table extraction
  - split_pages=False       → one Document per PDF

Supports batch loading: pass multiple file paths → single load() call to parse
all uploaded files together efficiently.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

from ingestion.models import ParsedDocument

logger = logging.getLogger("ingestion.loader")

# Element types that carry a plain-text "content" field
_TEXT_TYPES = {"paragraph", "heading", "caption", "list_item"}


def _extract_text(kids: Any) -> str:
    """
    Recursively collect every ``content`` field from text-bearing elements,
    returning them joined by double newlines.
    """
    if not isinstance(kids, list):
        return ""
    parts: list[str] = []
    for elem in kids:
        if isinstance(elem, str):
            if elem.strip():
                parts.append(elem.strip())
            continue
        if not isinstance(elem, dict):
            continue

        etype = elem.get("type", "")

        if etype in _TEXT_TYPES:
            content_val = elem.get("content")
            if isinstance(content_val, str) and content_val.strip():
                parts.append(content_val.strip())

        # Recurse into container / list elements
        for child_key in ("kids", "list_items"):
            child_val = elem.get(child_key)
            if isinstance(child_val, list):
                child_text = _extract_text(child_val)
                if child_text:
                    parts.append(child_text)

    return "\n\n".join(parts)


class PDFLoader:
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        logger.info("PDFLoader initialised (image_dir=%s)", image_dir)

    @staticmethod
    def parser(docs: list[Document]) -> list[ParsedDocument]:
        results: list[ParsedDocument] = []

        for doc in docs:
            raw: dict[str, Any] = {}
            page_content = getattr(doc, "page_content", {})

            if isinstance(page_content, str):
                try:
                    raw = json.loads(page_content)
                except Exception as exc:
                    logger.error("Failed to parse doc.page_content JSON: %s", exc)
                    raw = {}
            elif isinstance(page_content, dict):
                raw = page_content
            else:
                raw = {}

            if not isinstance(raw, dict):
                raw = {}

            kids: list[dict] = raw.get("kids", []) if isinstance(raw.get("kids"), list) else []
            plain_text = _extract_text(kids)

            meta = getattr(doc, "metadata", {}) or {}

            file_name = (
                raw.get("file_name")
                or meta.get("file_name")
                or (Path(meta.get("source", "")).name if meta.get("source") else None)
                or "document.pdf"
            )

            parsed = ParsedDocument(
                page_content=plain_text,
                metadata=meta,
                file_name=file_name,
                number_of_pages=raw.get("number_of_pages", 0) if isinstance(raw.get("number_of_pages"), int) else 0,
                author=raw.get("author") if isinstance(raw, dict) else None,
                title=raw.get("title") if isinstance(raw, dict) else None,
                creation_date=raw.get("creation_date") if isinstance(raw, dict) else None,
                modification_date=raw.get("modification_date") if isinstance(raw, dict) else None,
                kids=kids,
            )
            results.append(parsed)

        return results

    def load(self, file_paths: list[str]) -> list[ParsedDocument]:
        for path in file_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Batch loading %d PDF(s) via OpenDataLoaderPDFLoader …", len(file_paths))
        for fp in file_paths:
            logger.debug("  → %s (%s bytes)", fp, Path(fp).stat().st_size)

        fp_arg = file_paths[0] if len(file_paths) == 1 else file_paths
        raw_docs = OpenDataLoaderPDFLoader(
            file_path=fp_arg,
            format="json",
            image_output="external",
            image_dir=self.image_dir,
            image_format="png",
            table_method="cluster",
            split_pages=False,
            quiet=False,
        ).load()

        parsed = self.parser(raw_docs)

        logger.info(
            "Loaded %d document(s) — %d total pages",
            len(parsed),
            sum(d.number_of_pages for d in parsed),
        )
        return parsed
