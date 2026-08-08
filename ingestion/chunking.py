from typing import Any
from langchain_core.documents import Document
import json

class SemanticChunker:
    def __init__(self, max_chars: int, overlap_chars: int):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Document, user_id: str = "default") -> list[Document]:
        data: dict[str, Any] = json.loads(document.page_content)

        file_name = data.get("file name", "")
        docs: list[Document] = []

        print(f"Chunking document: {file_name} for user: {user_id}")

        self._visit(node=data, docs=docs, heading_stack=[], file_name=file_name, user_id=user_id)

        return self._merge_small_paragraphs(docs)

    def _visit(self, node: dict, docs: list[Document], heading_stack: list[str], file_name: str, user_id: str):
        node_type = node.get("type")

        if node_type in ("heading", "title"):
            heading = self._get_text(node)
            if heading:
                heading_stack = heading_stack + [heading]

        elif node_type in ("paragraph", "text_block", "list_item", "list item"):
            docs.extend(self._paragraph_to_docs(node, heading_stack, file_name, user_id))

        elif node_type == "table":
            docs.extend(self._table_to_docs(node, heading_stack, file_name, user_id))

        elif node_type in ("image", "figure"):
            doc = self._image_to_doc(node, heading_stack, file_name, user_id)

            if doc:
                docs.append(doc)

        elif node_type == "caption":
            docs.extend(self._paragraph_to_docs(node, heading_stack, file_name, user_id))

        for child in node.get("kids", []):
            self._visit(child, docs, heading_stack, file_name, user_id)

    @staticmethod
    def _make_chunk_id(user_id: str, file_name: str, raw_id: Any) -> str:
        raw_str = str(raw_id) if raw_id is not None else ""
        if raw_str.startswith(f"{user_id}_"):
            return raw_str
        if user_id and file_name and raw_str:
            return f"{user_id}_{file_name}_{raw_str}"
        elif user_id and file_name:
            return f"{user_id}_{file_name}"
        return raw_str

    def _get_text(self, node: dict) -> str:
        """
        OpenDataLoader versions have used either
        'content' or 'text'. Support both.
        """

        return (
            node.get("content")
            or node.get("text")
            or ""
        ).strip()

    def _paragraph_to_docs(self, node: dict, headings: list[str], file_name: str, user_id: str) -> list[Document]:
        text = self._get_text(node)

        if not text:
            return []

        prefix = "\n".join(filter(None, headings))

        full_text = f"{prefix}\n\n{text}" if prefix else text

        raw_id = node.get("id")
        chunk_id = self._make_chunk_id(user_id, file_name, raw_id)
        metadata = {
            "custom_id": chunk_id,
            "type": node.get("type"),
            "page": node.get("page number"),
            "bbox": node.get("bounding box"),
            "file_name": file_name,
        }

        raw_linked = node.get("linked content id")
        if raw_linked:
            metadata["linked_content_id"] = self._make_chunk_id(user_id, file_name, raw_linked)

        return self._split(
            full_text,
            metadata=metadata,
        )

    def _table_to_docs(self, node: dict, headings: list[str], file_name: str, user_id: str) -> list[Document]:

        rows = []

        for row in node.get("rows", []):
            cells = []
            for cell in row.get("cells", []):
                # Cell text lives inside its kids (paragraph nodes)
                cell_text = " ".join(
                    self._get_text(kid) for kid in cell.get("kids", [])
                    if self._get_text(kid)
                ).strip()
                if cell_text:
                    cells.append(cell_text)
            if cells:
                rows.append(" | ".join(cells))

        if not rows:
            return []

        table_text = "\n".join(rows)

        prefix = "\n".join(filter(None, headings))

        if prefix:
            table_text = prefix + "\n\n" + table_text

        raw_id = node.get("id")
        chunk_id = self._make_chunk_id(user_id, file_name, raw_id)
        metadata = {
            "custom_id": chunk_id,
            "type": "table",
            "page": node.get("page number"),
            "bbox": node.get("bounding box"),
            "file_name": file_name,
        }

        raw_linked = node.get("linked content id")
        if raw_linked:
            metadata["linked_content_id"] = self._make_chunk_id(user_id, file_name, raw_linked)

        return [
            Document(
                page_content=table_text,
                metadata=metadata,
            )
        ]


    def _image_to_doc(
        self,
        node: dict,
        headings: list[str],
        file_name: str,
        user_id: str,
    ):
        caption = (
            node.get("caption")
            or self._get_text(node)
            or "[image]"
        ).strip()

        prefix = "\n".join(filter(None, headings))

        text = f"{prefix}\n\n{caption}" if prefix else caption

        # OpenDataLoader stores base64 image under "data"; "source" may also exist
        image_path = node.get("data") or node.get("source") or node.get("alt_source")
        raw_id = node.get("id")
        chunk_id = self._make_chunk_id(user_id, file_name, raw_id)

        metadata = {
            "custom_id": chunk_id,
            "type": "image",
            "page": node.get("page number"),
            "bbox": node.get("bounding box"),
            "image_base64": image_path,
            "file_name": file_name,
        }

        raw_linked = node.get("linked content id")
        if raw_linked:
            metadata["linked_content_id"] = self._make_chunk_id(user_id, file_name, raw_linked)

        return Document(
            page_content=text,
            metadata=metadata,
        )

    def _split(
        self,
        text: str,
        metadata: dict,
    ) -> list[Document]:

        if len(text) <= self.max_chars:
            return [
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            ]

        docs = []
        start = 0

        while start < len(text):

            end = min(
                start + self.max_chars,
                len(text),
            )

            if end < len(text):

                while end > start and not text[end].isspace():
                    end -= 1

                if end == start:
                    end = min(
                        start + self.max_chars,
                        len(text),
                    )

            docs.append(
                Document(
                    page_content=text[start:end].strip(),
                    metadata=metadata.copy(),
                )
            )

            start = max(
                end - self.overlap_chars,
                start + 1,
            )

        return docs

    def _merge_small_paragraphs(
        self,
        docs: list[Document],
    ) -> list[Document]:

        merged = []
        buffer = None

        for doc in docs:

            if doc.metadata.get("type") != "paragraph":

                if buffer:
                    merged.append(buffer)
                    buffer = None

                merged.append(doc)
                continue

            if buffer is None:
                buffer = doc
                continue

            if (
                buffer.metadata["page"] == doc.metadata["page"]
                and len(buffer.page_content)
                + len(doc.page_content)
                + 2
                <= self.max_chars
            ):

                buffer.page_content += "\n\n" + doc.page_content

            else:
                merged.append(buffer)
                buffer = doc

        if buffer:
            merged.append(buffer)

        return merged