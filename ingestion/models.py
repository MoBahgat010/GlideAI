"""
Typed models for the OpenDataLoader PDF output.

``ParsedDocument`` extends LangChain's ``Document`` so it slots into any
LangChain chain or retriever transparently:

  - ``page_content``  — plain concatenated text (for chunking / embedding)
  - ``kids``          — full element tree (for image / table / caption extraction)
  - all other fields  — PDF-level metadata
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import Field
from langchain_core.documents import Document
from pydantic.dataclasses import dataclass

BoundingBox = tuple[float, float, float, float]  # [left, bottom, right, top]


@dataclass
class BaseElement:
    id: int
    level: str
    pdfua_tag: str
    page_number: int
    bounding_box: BoundingBox
    content: str

@dataclass
class Paragraph(BaseElement):
    type: Literal["paragraph"]

@dataclass
class Heading(BaseElement):
    type: Literal["heading"]

@dataclass
class Caption(BaseElement):
    type: Literal["caption"]
    linked_content_id: int

@dataclass
class Image(BaseElement):
    type: Literal["image"]
    source: str
    alt_source: str

@dataclass
class ListItem(BaseElement):
    type: Literal["list_item"]
    kids: list[BaseElement]

@dataclass
class List(BaseElement):
    type: Literal["list"]
    numbering_style: str
    number_of_list_items: int
    previous_list_id: int
    next_list_id: int
    list_items: list[ListItem]

@dataclass
class TableCell:
    type: Literal["table cell"]
    row_number: int
    column_number: int
    row_span: int
    column_span: int
    kids: list[BaseElement]

@dataclass
class TableRow:
    type: Literal["table row"]
    row_number: int
    cells: list[TableCell]

@dataclass
class Table(BaseElement):
    type: Literal["table"]
    number_of_rows: int
    number_of_columns: int
    previous_table_id: int
    next_table_id: int
    rows: list[TableRow]

@dataclass
class TextBlock(BaseElement):
    type: Literal["text_block"]
    kids: list[BaseElement]

@dataclass
class HeaderFooter(BaseElement):
    type: Literal["header", "footer"]
    kids: list[BaseElement]


class ParsedDocument(Document):
    """
    A :class:`langchain_core.documents.Document` enriched with PDF-level
    metadata and the raw element tree produced by OpenDataLoader.

    ``page_content`` holds the pre-extracted plain text (ready for embedding).
    ``kids`` holds the full element tree so the chunker can extract tables,
    images and captions without re-parsing.
    """

    # Pydantic v2: allow the extra fields below alongside LangChain's own fields
    model_config = {"arbitrary_types_allowed": True}

    # PDF-level metadata
    file_name: str = ""
    number_of_pages: int = 0
    author: Optional[str] = None
    title: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None

    # Raw element tree — kept for table / image / caption extraction
    kids: list[dict[str, Any]] = Field(default_factory=list)