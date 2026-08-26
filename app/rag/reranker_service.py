"""
Reranker Service.

High-level service for CrossEncoder document reranking.
"""

from __future__ import annotations

from app.core.config import reranker
from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument
from app.rag.reranker import DocumentReranker


class RerankerService:
    """
    High-level wrapper around DocumentReranker.
    """

    def __init__(self) -> None:
        logger.info(
            "Initializing RerankerService | model={}",
            reranker.model_name,
        )

        self._reranker = DocumentReranker()

        logger.info(
            "RerankerService initialized successfully."
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank documents according to their relevance
        to the supplied query.
        """

        if not documents:
            logger.warning(
                "Reranker received zero documents."
            )
            return []

        if not query or not query.strip():
            logger.warning(
                "Reranker received an empty query."
            )
            return documents

        logger.info(
            "Running CrossEncoder reranking | candidates={}",
            len(documents),
        )

        try:
            results = self._reranker.rerank(
                query=query.strip(),
                documents=documents,
            )

            logger.info(
                "CrossEncoder reranking completed | results={}",
                len(results),
            )

            return results

        except Exception:
            logger.exception(
                "CrossEncoder reranking failed."
            )

            # Safe fallback:
            # never destroy successful dense retrieval
            return documents