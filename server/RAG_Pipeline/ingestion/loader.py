import logging
from langchain_core.documents import Document
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

logger = logging.getLogger("ingestion.loader")

class PDFLoader:
    def __init__(self):
        logger.info("PDFLoader initialized with OpenDataLoaderPDFLoader")

    def load(self, folder_path: str) -> list[Document]:
        docs = OpenDataLoaderPDFLoader(
            file_path=folder_path,
            format="json",
            image_output="embedded",
            image_format="png",
            table_method="cluster",
            split_pages=False,
            quiet=True,
        ).load()

        logger.info("Loaded %s document(s)", folder_path)
        return docs
