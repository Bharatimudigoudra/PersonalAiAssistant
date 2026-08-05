"""
BM25 Service.

Provides a singleton interface for BM25 retrieval.
"""

from app.core.logging import logger
from app.rag.bm25.bm25 import BM25Retriever
from app.models.retrieved_document import RetrievedDocument

# ---------------------------------------------------------
# Singleton BM25 Retriever
# ---------------------------------------------------------

_bm25 = BM25Retriever()


class BM25Service:
    """
    High-level service for BM25 retrieval.
    """

    def __init__(self) -> None:

        self._retriever = _bm25

        logger.info(
            "BM25Service initialized."
        )

    def build_index(
        self,
        documents: list[RetrievedDocument],
    ) -> None:
        """
        Build the BM25 index.
        """

        logger.info(
            "Building BM25 index..."
        )

        self._retriever.build_index(
            documents,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        """
        Search the BM25 index.
        """

        logger.info(
            "Running BM25 retrieval..."
        )

        return self._retriever.search(
            query=query,
            top_k=top_k,
        )


# ---------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------

_bm25_service = BM25Service()


def get_bm25_service() -> BM25Service:
    """
    Return singleton BM25 service.
    """

    return _bm25_service