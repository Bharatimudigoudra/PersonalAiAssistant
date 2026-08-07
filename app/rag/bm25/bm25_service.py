"""
BM25 Service.

High-level singleton wrapper around the BM25Retriever.
"""

from __future__ import annotations

from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument
from app.rag.bm25.bm25 import BM25Retriever


class BM25Service:
    """
    Singleton service that manages the BM25 index.
    """

    def __init__(self) -> None:

        self._retriever = BM25Retriever()

        logger.info(
            "BM25Service initialized."
        )

    # ---------------------------------------------------------
    # Index Management
    # ---------------------------------------------------------

    def build_index(
        self,
        documents: list[RetrievedDocument],
    ) -> None:
        """
        Build or rebuild the BM25 index.
        """

        if not documents:

            logger.warning(
                "No documents supplied for BM25 indexing."
            )

            return

        logger.info(
            "Building BM25 index with {} documents...",
            len(documents),
        )

        self._retriever.build_index(
            documents,
        )

        logger.info(
            "BM25 index successfully built."
        )

    def has_index(
        self,
    ) -> bool:
        """
        Returns True if a BM25 index exists.
        """

        return getattr(
            self._retriever,
            "_documents",
            None,
        ) not in (
            None,
            [],
        )

    def clear(
        self,
    ) -> None:
        """
        Reset the retriever.
        """

        self._retriever = BM25Retriever()

        logger.info(
            "BM25 index cleared."
        )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        """
        Execute BM25 retrieval.
        """

        if not self.has_index():

            logger.warning(
                "BM25 index is empty."
            )

            return []

        logger.info(
            "Running BM25 retrieval (top_k={})",
            top_k,
        )

        results = self._retriever.search(
            query=query,
            top_k=top_k,
        )

        logger.info(
            "BM25 returned {} documents.",
            len(results),
        )

        return results


# ---------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------

_bm25_service = BM25Service()


def get_bm25_service() -> BM25Service:
    """
    Returns the global BM25 service.
    """

    return _bm25_service