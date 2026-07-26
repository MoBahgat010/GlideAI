"""
Document chunking for the multimodal RAG pipeline.

Processes OpenDataLoader JSON output into three types of indexed records:

1. **Text records** — semantically chunked consecutive text elements
   (paragraphs, headings, list items, captions, table text).
   Embedded via jina-clip-v2 text tower.

2. **Image records** — extracted images saved to disk.
   Embedded via jina-clip-v2 image tower.

3. **Caption records** — text paired with an image (the nearest caption
   or surrounding paragraph).
   Embedded via jina-clip-v2 text tower, cross-referenced to the image
   record via ``linked_image_id``.

All record IDs are **deterministic** (SHA-256 of source_key + type + index)
so re-ingesting the same document is idempotent.
"""

import base64
import hashlib
import io
import json
import logging
from pathlib import Path

from PIL import Image
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger("ingestion.chunking")

# Element types that contribute running text
_TEXT_TYPES = {
    "paragraph",
    "heading",
    "caption",
    "list item",
    "list_item",
    "footnote",
}


def _deterministic_id(source_key: str, record_type: str, index: int | str) -> str:
    """SHA-256 based deterministic ID for idempotent upserts."""
    raw = f"{source_key}:{record_type}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _extract_table_text(element: dict) -> str:
    """Extract readable text from a table element."""
    rows = element.get("rows", [])
    if not rows:
        return element.get("content", "")
    lines = []
    for row in rows:
        cells = row if isinstance(row, list) else row.get("cells", [])
        cell_texts = []
        for cell in cells:
            if isinstance(cell, str):
                cell_texts.append(cell)
            elif isinstance(cell, dict):
                cell_texts.append(cell.get("content", cell.get("text", "")))
        lines.append(" | ".join(cell_texts))
    return "\n".join(lines)


class DocumentChunker:
    """
    Processes OpenDataLoader JSON output into indexed records
    (text, image, caption) for the multimodal Pinecone index.

    Parameters
    ----------
    embeddings : Embeddings
        LangChain Embeddings object (JinaClipTextEmbeddings adapter)
        used by SemanticChunker for breakpoint detection.
    image_dir : str
        Base directory where extracted images are saved to disk.
    """

    def __init__(self, embeddings: Embeddings, image_dir: str):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def process_document(
        self,
        parsed_json: dict | str,
        source_key: str,
        job_id: str,
        page_offset: int = 0,
    ) -> list[dict]:
        """
        Walk the parsed JSON structure and produce records.

        Parameters
        ----------
        parsed_json : dict | str
            OpenDataLoader JSON output (as a dict or a JSON string).
        source_key : str
            Storage key of the original file (e.g. ``"report.pdf"``).
        job_id : str
            Ingestion job identifier.
        page_offset : int
            Offset to add to page numbers (used when parsing PDFs in chunks).

        Returns
        -------
        list[dict]
            Records, each with keys:
              id, record_type, text, image (PIL.Image or None),
              image_path (str or None), metadata (dict)
        """
        if isinstance(parsed_json, str):
            parsed_json = json.loads(parsed_json)

        filename = parsed_json.get("file name", Path(source_key).name)
        elements = parsed_json.get("kids", [])
        logger.info(
            "Processing %s — %d top-level elements", source_key, len(elements)
        )

        # Flatten nested elements (some structures nest kids within kids)
        flat_elements = self._flatten_elements(elements)
        logger.info("Flattened to %d elements", len(flat_elements))

        # ── Separate text and image elements ──────────────────────────────────
        text_segments: list[dict] = []  # {text, page_number, bbox, type}
        image_elements: list[dict] = []

        for el in flat_elements:
            el_type = el.get("type", "").lower().strip()

            if el_type in ("image", "figure"):
                el["_page_offset"] = page_offset
                image_elements.append(el)
            elif el_type == "table":
                table_text = _extract_table_text(el)
                if table_text.strip():
                    text_segments.append({
                        "text": table_text,
                        "page_number": el.get("page number", el.get("page_number", 0)) + page_offset,
                        "bbox": el.get("bounding box", el.get("bbox", [])),
                        "type": "table",
                    })
            elif el_type in _TEXT_TYPES or el.get("content", "").strip():
                content = el.get("content", el.get("text", "")).strip()
                if content:
                    text_segments.append({
                        "text": content,
                        "page_number": el.get("page number", el.get("page_number", 0)) + page_offset,
                        "bbox": el.get("bounding box", el.get("bbox", [])),
                        "type": el_type,
                    })

        logger.info(
            "%d text segments, %d image elements",
            len(text_segments), len(image_elements),
        )

        records: list[dict] = []

        # ── 1. Semantic-chunk text segments ───────────────────────────────────
        if text_segments:
            text_records = self._chunk_text_segments(
                text_segments, source_key, filename, job_id
            )
            records.extend(text_records)

        # ── 2. Process images ─────────────────────────────────────────────────
        if image_elements:
            img_records = self._process_images(
                image_elements, text_segments, source_key, filename, job_id
            )
            records.extend(img_records)

        logger.info(
            "Produced %d total records for %s (text=%d, image/caption=%d)",
            len(records), source_key,
            sum(1 for r in records if r["record_type"] == "text"),
            sum(1 for r in records if r["record_type"] in ("image", "caption")),
        )
        return records

    # ── private: text chunking ────────────────────────────────────────────────

    def _chunk_text_segments(
        self,
        segments: list[dict],
        source_key: str,
        filename: str,
        job_id: str,
    ) -> list[dict]:
        """
        Concatenate text segments and run RecursiveCharacterTextSplitter,
        then map each chunk back to its source pages and bounding boxes.
        """
        full_text = "\n\n".join(seg["text"] for seg in segments)
        if len(full_text.strip()) < 20:
            logger.warning("Extracted text is nearly empty for %s", source_key)
            return []

        lc_docs = self.splitter.create_documents([full_text])
        logger.debug("Recursive text splitter produced %d final chunks", len(lc_docs))

        records = []
        # Build a mapping from character offsets to segment metadata
        seg_boundaries = self._build_segment_boundaries(segments)

        for i, doc in enumerate(lc_docs):
            chunk_text = doc.page_content
            # Find which pages/bboxes this chunk spans
            page_numbers, bboxes = self._find_span_metadata(
                chunk_text, full_text, seg_boundaries
            )

            record_id = _deterministic_id(source_key, "text", i)
            records.append({
                "id": record_id,
                "record_type": "text",
                "text": chunk_text,
                "image": None,
                "image_path": None,
                "metadata": {
                    "type": "text",
                    "filename": filename,
                    "source_key": source_key,
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                    "page_numbers": page_numbers,
                    "bboxes": bboxes,
                    "ingestion_job_id": job_id,
                    "chunk_text": chunk_text[:1000],
                    "linked_image_id": None,
                    "linked_text_id": None,
                },
            })

        logger.info("Text chunking → %d records", len(records))
        for j, r in enumerate(records[:3]):
            logger.debug(
                "  chunk[%d]: %d chars, pages=%s — %r",
                j, r["metadata"]["char_count"],
                r["metadata"]["page_numbers"],
                r["text"][:100],
            )
        return records

    # ── private: image processing ─────────────────────────────────────────────

    def _process_images(
        self,
        image_elements: list[dict],
        text_segments: list[dict],
        source_key: str,
        filename: str,
        job_id: str,
    ) -> list[dict]:
        """
        Save image data to disk, extract captions, and build records.
        """
        records = []
        doc_image_dir = self.image_dir / job_id
        doc_image_dir.mkdir(parents=True, exist_ok=True)

        for idx, el in enumerate(image_elements):
            page_offset = el.pop("_page_offset", 0)
            page_num = el.get("page number", el.get("page_number", 0)) + page_offset
            bbox = el.get("bounding box", el.get("bbox", []))

            # ── Decode and save image ─────────────────────────────────────────
            image_data = (
                el.pop("image_data", None)
                or el.pop("image", None)
                or el.pop("content", None)
            )
            if not image_data:
                logger.warning(
                    "Image element on page %d has no image data — skipping",
                    page_num,
                )
                continue

            # Check if it's already a file path (from image_output="directory")
            if isinstance(image_data, str) and Path(image_data).exists():
                image_path = Path(image_data)
                logger.debug("Image already on disk → %s", image_path)
            else:
                # Strip data-URI prefix if present
                if "base64," in image_data:
                    image_data = image_data.split("base64,", 1)[1]

                try:
                    raw_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    
                    image_filename = f"p{page_num}_{idx}.png"
                    image_path = doc_image_dir / image_filename
                    pil_image.save(str(image_path), "PNG")
                    logger.debug("Saved image → %s (%dx%d)", image_path, *pil_image.size)
                    del pil_image  # Free RAM early
                except Exception:
                    logger.warning(
                        "Failed to decode image on page %d — skipping", page_num
                    )
                    continue

            # ── Find caption ──────────────────────────────────────────────────
            caption = self._find_caption(el, text_segments, page_num)

            # ── Image record ──────────────────────────────────────────────────
            image_id = _deterministic_id(source_key, "image", f"{page_num}_{idx}")
            caption_id = _deterministic_id(source_key, "caption", f"{page_num}_{idx}")

            records.append({
                "id": image_id,
                "record_type": "image",
                "text": None,
                "image": None,
                "image_path": str(image_path),
                "metadata": {
                    "type": "image",
                    "filename": filename,
                    "source_key": source_key,
                    "page_numbers": [page_num],
                    "bboxes": [bbox] if bbox else [],
                    "image_path": str(image_path),
                    "caption": caption[:500] if caption else "",
                    "ingestion_job_id": job_id,
                    "chunk_text": caption[:1000] if caption else "",
                    "linked_image_id": None,
                    "linked_text_id": caption_id if caption else None,
                },
            })

            # ── Caption record (only if caption text exists) ──────────────────
            if caption and caption.strip():
                records.append({
                    "id": caption_id,
                    "record_type": "caption",
                    "text": caption,
                    "image": None,
                    "image_path": None,
                    "metadata": {
                        "type": "caption",
                        "filename": filename,
                        "source_key": source_key,
                        "page_numbers": [page_num],
                        "bboxes": [bbox] if bbox else [],
                        "ingestion_job_id": job_id,
                        "chunk_text": caption[:1000],
                        "linked_image_id": image_id,
                        "linked_text_id": None,
                    },
                })

        logger.info(
            "Image processing → %d records (%d images, %d captions)",
            len(records),
            sum(1 for r in records if r["record_type"] == "image"),
            sum(1 for r in records if r["record_type"] == "caption"),
        )
        return records

    # ── private: helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _flatten_elements(elements: list) -> list[dict]:
        """Recursively flatten nested 'kids' structures."""
        flat = []
        for el in elements:
            if isinstance(el, dict):
                flat.append(el)
                if "kids" in el:
                    flat.extend(DocumentChunker._flatten_elements(el["kids"]))
        return flat

    @staticmethod
    def _build_segment_boundaries(
        segments: list[dict],
    ) -> list[tuple[int, int, dict]]:
        """
        Build (start_offset, end_offset, segment_metadata) tuples
        corresponding to the concatenated full_text.
        """
        boundaries = []
        offset = 0
        for seg in segments:
            text = seg["text"]
            start = offset
            end = offset + len(text)
            boundaries.append((start, end, seg))
            offset = end + 2  # +2 for the "\n\n" separator
        return boundaries

    @staticmethod
    def _find_span_metadata(
        chunk_text: str,
        full_text: str,
        seg_boundaries: list[tuple[int, int, dict]],
    ) -> tuple[list[int], list[list]]:
        """
        Given a chunk's text, find it in full_text and return
        which pages and bounding boxes it spans.
        """
        idx = full_text.find(chunk_text)
        if idx < 0:
            # Fallback: try first 200 chars
            idx = full_text.find(chunk_text[:200])
        if idx < 0:
            return [], []

        chunk_start = idx
        chunk_end = idx + len(chunk_text)

        pages: list[int] = []
        bboxes: list[list] = []
        for seg_start, seg_end, seg_meta in seg_boundaries:
            if seg_start < chunk_end and seg_end > chunk_start:
                page = seg_meta.get("page_number", 0)
                if page and page not in pages:
                    pages.append(page)
                bbox = seg_meta.get("bbox", [])
                if bbox:
                    bboxes.append(bbox)
        return pages, bboxes

    @staticmethod
    def _find_caption(
        image_element: dict,
        text_segments: list[dict],
        page_num: int,
    ) -> str:
        """
        Find the best caption for an image by looking for:
        1. A 'caption' type element on the same page
        2. The linked_content_id reference
        3. The nearest paragraph on the same page
        """
        linked_id = image_element.get("linked content id")

        # Strategy 1: Look for caption elements on same page
        same_page = [
            s for s in text_segments
            if s.get("page_number") == page_num and s.get("type") == "caption"
        ]
        if same_page:
            return same_page[0]["text"]

        # Strategy 2: linked_content_id (if available, already captured above)
        if linked_id is not None:
            for seg in text_segments:
                if seg.get("id") == linked_id:
                    return seg["text"]

        # Strategy 3: nearest paragraph on same page
        same_page_paras = [
            s for s in text_segments
            if s.get("page_number") == page_num and s.get("type") == "paragraph"
        ]
        if same_page_paras:
            return same_page_paras[-1]["text"]

        return ""
