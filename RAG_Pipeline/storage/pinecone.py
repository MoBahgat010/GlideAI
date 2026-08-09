"""
Backwards-compatibility alias for WeaviateVDB.
"""

import logging
from RAG_Pipeline.storage.weaviate import WeaviateVDB
import config
logger = logging.getLogger("storage.pinecone")


class PineconeVDB(WeaviateVDB):
    """
    Deprecated alias: redirects PineconeVDB calls to WeaviateVDB.
    """

    def __init__(self, api_key: str, index_name: str, dimension: int, **kwargs):
        endpoint = getattr(config, "WEAVIATE_REST_ENDPOINT", "")
        w_api_key = getattr(config, "WEAVIATE_API_KEY", "") or api_key
        logger.warning("PineconeVDB is deprecated; redirecting to WeaviateVDB (endpoint=%s)", endpoint)
        super().__init__(endpoint=endpoint, api_key=w_api_key, index_name=index_name, dimension=dimension)
