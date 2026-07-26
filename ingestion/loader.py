"""
PDF loading via OpenDataLoaderPDFLoader (LangChain integration).

Wraps the loader with the exact configuration for the multimodal RAG
pipeline:
  - format="json"        → element-level structure (type, bbox, page)
  - image_output="embedded" → base64 images inline
  - image_format="png"
  - table_method="cluster"  → AI-assisted table extraction
  - hybrid="docling-fast", hybrid_mode="auto" → routes complex pages
                                                 to AI backend
  - quiet=True

Supports batch loading: pass multiple file paths → single load() call.
"""

import logging
from pathlib import Path

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

logger = logging.getLogger("ingestion.loader")


class PDFLoader:
    """
    Batch PDF loader using OpenDataLoaderPDFLoader.

    Accepts a list of file paths (or a single path) and returns
    LangChain Document objects with JSON-formatted page_content
    and metadata (page_number, source, etc.).
    """

    def __init__(self, hybrid_url: str = "http://localhost:5002", image_dir: str | None = None):
        self.hybrid_url = hybrid_url
        self.image_dir = image_dir
        if self.image_dir:
            Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        logger.info("PDFLoader initialised (hybrid_url=%s, image_dir=%s)", hybrid_url, image_dir)

    def load(self, file_paths: list[str] | str) -> list:
        """
        Load one or more PDFs and return LangChain Documents.

        Uses OpenDataLoader batching: a single loader instance
        processes all files in one call, minimising Java process
        overhead.

        Parameters
        ----------
        file_paths : list[str] | str
            Path(s) to PDF files.

        Returns
        -------
        list[Document]
            LangChain Documents. With format="json" and split_pages=False,
            each document corresponds to one input PDF, with the full
            JSON structure as page_content.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Validate all paths exist
        for fp in file_paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"PDF not found: {fp}")

        logger.info("Loading %d PDF(s) via OpenDataLoaderPDFLoader …", len(file_paths))
        for fp in file_paths:
            logger.debug("  → %s (%s bytes)", fp, Path(fp).stat().st_size)

        loader = OpenDataLoaderPDFLoader(
            file_path=file_paths,
            format="json",
            image_output="external" if self.image_dir else "embedded",
            image_dir=self.image_dir,
            image_format="png",
            table_method="cluster",
            # hybrid="docling-fast",
            hybrid_mode="auto",
            # hybrid_url=self.hybrid_url,
            split_pages=False,
            quiet=False,
        )

        documents = loader.load()

        logger.info(
            "Loaded %d document(s) — total page_content chars: %s",
            len(documents),
            sum(len(d.page_content) for d in documents),
        )
        return documents
