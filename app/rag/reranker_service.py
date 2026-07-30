"""
Reranker service.

Provides a singleton interface for CrossEncoder document reranking.
"""

from app.core.logging import logger
from app.rag.reranker import DocumentReranker
from app.rag.retrieval import RetrievedDocument

# ---------------------------------------------------------------------
# Singleton Reranker
# ---------------------------------------------------------------------

_reranker = DocumentReranker()


class RerankerService:
    """
    High-level service for reranking retrieved documents.
    """

    def __init__(self) -> None:

        self.reranker = _reranker

        logger.info(
            "RerankerService initialized."
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank retrieved documents using the CrossEncoder model.
        """

        logger.info(
            "Running CrossEncoder reranker..."
        )

        reranked_documents = self.reranker.rerank(
            query=query,
            documents=documents,
        )

        logger.info(
            "Reranking completed. {} documents returned.",
            len(reranked_documents),
        )

        return reranked_documents