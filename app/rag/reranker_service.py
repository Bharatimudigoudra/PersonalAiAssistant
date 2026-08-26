"""
Reranker service.

Provides a singleton interface for CrossEncoder document reranking.
"""

from __future__ import annotations

from app.core.config import reranker
from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument
from app.rag.reranker import DocumentReranker


class RerankerService:
    """
    High-level service for CrossEncoder document reranking.

    The underlying CrossEncoder model is loaded only once and
    shared across the application.
    """

    _instance: "RerankerService | None" = None

    def __new__(cls) -> "RerankerService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        logger.info(
            "Initializing RerankerService | model={}",
            reranker.model_name,
        )

        self.reranker = DocumentReranker()

        self._initialized = True

        logger.info(
            "RerankerService initialized successfully."
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank documents against a query.
        """

        if not query or not query.strip():
            logger.warning(
                "Reranker received an empty query."
            )
            return documents

        if not documents:
            logger.warning(
                "Reranker received no documents."
            )
            return []

        try:
            logger.info(
                "Running CrossEncoder reranker | candidates={}",
                len(documents),
            )

            results = self.reranker.rerank(
                query=query.strip(),
                documents=documents,
            )

            logger.info(
                "Reranking completed | returned={}",
                len(results),
            )

            return results

        except Exception:
            logger.exception(
                "Reranking failed. Returning original candidates."
            )

            return documents

    def health_check(self) -> bool:
        """
        Verify that the reranker model is available.
        """

        try:
            if self.reranker.model is None:
                return False

            # Small inference test.
            scores = self.reranker.model.predict(
                [
                    (
                        "test query",
                        "test document",
                    )
                ],
                show_progress_bar=False,
            )

            return scores is not None and len(scores) > 0

        except Exception:
            logger.exception(
                "Reranker health check failed."
            )
            return False


def get_reranker_service() -> RerankerService:
    """
    Return the singleton RerankerService.
    """

    return RerankerService()