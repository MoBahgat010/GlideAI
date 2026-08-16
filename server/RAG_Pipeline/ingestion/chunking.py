import json
import logging
from typing import Any, Optional
from langchain_core.documents import Document
import json

logger = logging.getLogger("ingestion.chunking")

class SemanticChunker:
    def __init__(self, max_chars: int = 1000, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars


    def chunk(self, document: Document, session_id: str) -> list[Document]:
        data: dict[str, Any] = json.loads(document.page_content)

        file_name = data.get("file name", "")
        docs: list[Document] = []
        logger.info("Chunking document '%s' for session '%s'", file_name, session_id)

        for element in data.get("kids", []):
            self._visit(
                node=element,
                docs=docs,
                heading_stack=[],
                file_name=file_name,
                session_id=session_id,
            )

        merged = self._merge_small_paragraphs(docs)
        file_url = document.metadata.get("file_url")
        if file_url:
            for d in merged:
                d.metadata["file_url"] = file_url
        return merged


    def _visit(
        self,
        node: dict,
        docs: list[Document],
        heading_stack: list[str],
        file_name: str,
        session_id: str,
    ) -> None:
        node_type = node.get("type", "")

        if node_type in {"header", "footer"}:
            return

        if node_type == "heading":
            heading = self._get_content(node)
            if heading:
                heading_stack = heading_stack + [heading]
            for kid in node.get("kids", []):
                self._visit(kid, docs, heading_stack, file_name, session_id)
            return

        if node_type in ("paragraph", "caption"):
            docs.extend(
                self._text_node_to_docs(node, heading_stack, file_name, session_id)
            )
            return

        if node_type == "transcript":
            docs.extend(
                self._transcript_node_to_docs(node, heading_stack, file_name, session_id)
            )
            return

        if node_type == "text block":
            for kid in node.get("kids", []):
                self._visit(kid, docs, heading_stack, file_name, session_id)
            return

        if node_type == "list":
            self._visit_list(node, docs, heading_stack, file_name, session_id)
            return

        if node_type == "list item":
            self._visit_list_item(node, docs, heading_stack, file_name, session_id)
            return

        if node_type == "table":
            docs.extend(
                self._table_to_docs(node, heading_stack, file_name, session_id)
            )
            return

        if node_type in ("image", "figure"):
            doc = self._image_to_doc(node, heading_stack, file_name, session_id)
            if doc:
                docs.append(doc)
            return

        for kid in node.get("kids", []):
            self._visit(kid, docs, heading_stack, file_name, session_id)

    def _transcript_node_to_docs(
        self,
        node: dict,
        headings: list[str],
        file_name: str,
        session_id: str,
    ) -> list[Document]:
        text = self._get_content(node)
        if not text:
            return []

        prefix = "\n".join(filter(None, headings))
        full_text = f"{prefix}\n\n{text}" if prefix else text

        start_time = float(node.get("start_time", 0.0))
        end_time = float(node.get("end_time", start_time))
        raw_id = f"ts_{int(start_time)}_{int(end_time)}"
        chunk_id = self._make_chunk_id(session_id, file_name, raw_id)

        metadata: dict = {
            "custom_id": chunk_id,
            "session_id": session_id,
            "type": "transcript",
            "page": node.get("page number", 1),
            "start_time": start_time,
            "end_time": end_time,
            "file_name": file_name,
        }

        return self._split(full_text, metadata=metadata)

    def _visit_list(
        self,
        node: dict,
        docs: list[Document],
        heading_stack: list[str],
        file_name: str,
        session_id: str,
    ) -> None:
        list_items: list[dict] = node.get("list items", [])
        numbering_style: str = node.get("numbering style", "")

        item_lines: list[str] = []
        nested_nodes: list[dict] = []

        for item in list_items:
            if item.get("type") != "list item":
                nested_nodes.append(item)
                continue

            item_text = self._get_content(item)

            for kid in item.get("kids", []):
                kid_type = kid.get("type", "")
                if kid_type == "list":
                    nested_nodes.append(kid)
                elif kid_type in ("paragraph", "text block"):
                    extra = self._get_content(kid) or self._extract_kids_text(kid)
                    if extra:
                        item_text = (item_text + " " + extra).strip()
                else:
                    nested_nodes.append(kid)

            if item_text:
                item_lines.append(item_text)

        if item_lines:
            prefix = "\n".join(filter(None, heading_stack))
            list_text = "\n".join(f"• {line}" for line in item_lines)
            full_text = f"{prefix}\n\n{list_text}" if prefix else list_text

            raw_id = node.get("id")
            chunk_id = self._make_chunk_id(session_id, file_name, raw_id)

            metadata: dict = {
                "custom_id": chunk_id,
                "session_id": session_id,
                "type": "list",
                "numbering_style": numbering_style,
                "page": node.get("page number"),
                "bbox": node.get("bounding box"),
                "file_name": file_name,
            }

            if node.get("previous list id") is not None:
                metadata["previous_list_id"] = self._make_chunk_id(
                    session_id, file_name, node["previous list id"]
                )
            if node.get("next list id") is not None:
                metadata["next_list_id"] = self._make_chunk_id(
                    session_id, file_name, node["next list id"]
                )

            docs.extend(self._split(full_text, metadata=metadata))

        for nested in nested_nodes:
            self._visit(nested, docs, heading_stack, file_name, session_id)

    def _visit_list_item(
        self,
        node: dict,
        docs: list[Document],
        heading_stack: list[str],
        file_name: str,
        session_id: str,
    ) -> None:
        item_text = self._get_content(node)
        nested_nodes: list[dict] = []

        for kid in node.get("kids", []):
            kid_type = kid.get("type", "")
            if kid_type == "list":
                nested_nodes.append(kid)
            elif kid_type in ("paragraph", "text block"):
                extra = self._get_content(kid) or self._extract_kids_text(kid)
                if extra:
                    item_text = (item_text + " " + extra).strip()
            else:
                nested_nodes.append(kid)

        if item_text:
            docs.extend(
                self._text_node_to_docs(
                    node,
                    heading_stack,
                    file_name,
                    session_id,
                    override_text=item_text,
                )
            )

        for nested in nested_nodes:
            self._visit(nested, docs, heading_stack, file_name, session_id)

    def _text_node_to_docs(
        self,
        node: dict,
        headings: list[str],
        file_name: str,
        session_id: str,
        override_text: str | None = None,
    ) -> list[Document]:
        text = override_text if override_text is not None else self._get_content(node)

        if not text:
            return []

        prefix = "\n".join(filter(None, headings))
        full_text = f"{prefix}\n\n{text}" if prefix else text

        raw_id = node.get("id")
        chunk_id = self._make_chunk_id(session_id, file_name, raw_id)
        metadata: dict = {
            "custom_id": chunk_id,
            "session_id": session_id,
            "type": node.get("type", "paragraph"),
            "page": node.get("page number"),
            "bbox": node.get("bounding box"),
            "file_name": file_name,
        }

        raw_linked = node.get("linked content id")
        if raw_linked is not None:
            metadata["linked_content_id"] = self._make_chunk_id(
                session_id, file_name, raw_linked
            )

        return self._split(full_text, metadata=metadata)

    def _table_to_docs(
        self,
        node: dict,
        headings: list[str],
        file_name: str,
        session_id: str,
    ) -> list[Document]:
        rows_text: list[str] = []

        for row in node.get("rows", []):
            cells: list[str] = []
            for cell in row.get("cells", []):
                cell_text = self._extract_kids_text(cell)
                if cell_text:
                    cells.append(cell_text)
            if cells:
                rows_text.append(" | ".join(cells))

        if not rows_text:
            return []

        table_text = "\n".join(rows_text)
        prefix = "\n".join(filter(None, headings))
        if prefix:
            table_text = prefix + "\n\n" + table_text

        raw_id = node.get("id")
        chunk_id = self._make_chunk_id(session_id, file_name, raw_id)
        metadata: dict = {
            "custom_id": chunk_id,
            "session_id": session_id,
            "type": "table",
            "page": node.get("page number"),
            "bbox": node.get("bounding box"),
            "file_name": file_name,
        }

        if node.get("previous table id") is not None:
            metadata["previous_table_id"] = self._make_chunk_id(
                session_id, file_name, node["previous table id"]
            )
        if node.get("next table id") is not None:
            metadata["next_table_id"] = self._make_chunk_id(
                session_id, file_name, node["next table id"]
            )

        raw_linked = node.get("linked content id")
        if raw_linked is not None:
            metadata["linked_content_id"] = self._make_chunk_id(
                session_id, file_name, raw_linked
            )

        return [Document(page_content=table_text, metadata=metadata)]

    def _image_to_doc(
        self,
        node: dict,
        headings: list[str],
        file_name: str,
        session_id: str,
    ) -> Optional[Document]:
        image_ref = node.get("data", None)
        if not image_ref:
            return None

        raw_id = node.get("id")
        chunk_id = self._make_chunk_id(session_id, file_name, raw_id)

        metadata: dict = {
            "custom_id": chunk_id,
            "session_id": session_id,
            "type": "image",
            "page": node.get("page number"),
            "bbox": node.get("bounding box"),
            "file_name": file_name,
            "image_base64": image_ref,
        }

        raw_linked = node.get("linked content id")
        if raw_linked is not None:
            metadata["linked_content_id"] = self._make_chunk_id(
                session_id, file_name, raw_linked
            )

        return Document(page_content=image_ref, metadata=metadata)


    def _extract_kids_text(self, node: dict) -> str:
        parts: list[str] = []
        for kid in node.get("kids", []):
            text = self._get_content(kid)
            if text:
                parts.append(text)
            if kid.get("kids"):
                nested = self._extract_kids_text(kid)
                if nested:
                    parts.append(nested)
        return " ".join(parts).strip()

    @staticmethod
    def _get_content(node: dict) -> str:
        return (node.get("content") or node.get("text") or "").strip()

    @staticmethod
    def _make_chunk_id(session_id: str, file_name: str, raw_id: Any) -> str:
        raw_str = str(raw_id) if raw_id is not None else ""
        if raw_str.startswith(f"{session_id}_"):
            return raw_str
        if session_id and file_name and raw_str:
            return f"{session_id}_{file_name}_{raw_str}"
        elif session_id and file_name:
            return f"{session_id}_{file_name}"
        return raw_str

    def _split(self, text: str, metadata: dict) -> list[Document]:
        if len(text) <= self.max_chars:
            return [Document(page_content=text, metadata=metadata)]

        docs: list[Document] = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chars, len(text))

            if end < len(text):
                tmp = end
                while tmp > start and not text[tmp].isspace():
                    tmp -= 1
                if tmp > start:
                    end = tmp

            docs.append(
                Document(
                    page_content=text[start:end].strip(),
                    metadata=metadata.copy(),
                )
            )

            start = max(end - self.overlap_chars, start + 1)

        return docs


    def _merge_small_paragraphs(self, docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        buffer: Document | None = None

        for doc in docs:
            doc_type = doc.metadata.get("type")
            is_mergeable = doc_type in {"paragraph", "list item", "text block", "transcript"}

            if not is_mergeable:
                if buffer is not None:
                    merged.append(buffer)
                    buffer = None
                merged.append(doc)
                continue

            if buffer is None:
                buffer = doc
                continue

            same_page = buffer.metadata.get("page") == doc.metadata.get("page")
            fits = (
                len(buffer.page_content) + len(doc.page_content) + 2
                <= self.max_chars
            )

            if same_page and fits:
                buffer.page_content += "\n\n" + doc.page_content
            else:
                merged.append(buffer)
                buffer = doc

        if buffer is not None:
            merged.append(buffer)

        return merged
