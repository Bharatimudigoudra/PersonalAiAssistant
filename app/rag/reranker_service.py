"""
Cross-Encoder document reranker.

Loads the reranker model lazily and assigns reranker scores
to RetrievedDocument objects.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import reranker
from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument


class DocumentReranker:
    """
    Cross-Encoder based document reranker.

    The model is loaded lazily so importing this module does not
    immediately download/load the model.
    """

    def __init__(self) -> None:
        self.model: CrossEncoder | None = None
        self.model_name = reranker.model_name

        logger.info(
            "DocumentReranker initialized | model={}",
            self.model_name,
        )

    def _load_model(self) -> CrossEncoder:
        """
        Load the CrossEncoder model only when required.
        """

        if self.model is None:
            logger.info(
                "Loading reranker model: {}",
                self.model_name,
            )

            self.model = CrossEncoder(
                self.model_name,
            )

            logger.info(
                "Reranker model loaded successfully | model={}",
                self.model_name,
            )

        return self.model

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Rerank documents using a CrossEncoder.

        Args:
            query:
                User/interview query.

            documents:
                Candidate documents from retrieval.

            top_k:
                Number of documents to return.

        Returns:
            Reranked RetrievedDocument objects.
        """

        if not query or not query.strip():
            logger.warning(
                "Cannot rerank with an empty query."
            )
            return []

        if not documents:
            logger.warning(
                "No documents available for reranking."
            )
            return []

        query = query.strip()

        # ---------------------------------------------------------
        # Remove invalid documents
        # ---------------------------------------------------------

        valid_documents: list[RetrievedDocument] = []

        for document in documents:
            if document is None:
                continue

            content = str(
                document.content or ""
            ).strip()

            if not content:
                continue

            valid_documents.append(document)

        if not valid_documents:
            logger.warning(
                "No valid documents available for reranking."
            )
            return []

        # ---------------------------------------------------------
        # Resolve top_k
        # ---------------------------------------------------------

        if top_k is None:
            top_k = int(
                getattr(
                    reranker,
                    "top_k",
                    3,
                )
            )

        top_k = max(1, int(top_k))

        top_k = min(
            top_k,
            len(valid_documents),
        )

        # ---------------------------------------------------------
        # Load model lazily
        # ---------------------------------------------------------

        model = self._load_model()

        # ---------------------------------------------------------
        # Create query/document pairs
        # ---------------------------------------------------------

        pairs = [
            (
                query,
                document.content,
            )
            for document in valid_documents
        ]

        logger.info(
            "Reranking {} documents | top_k={}",
            len(pairs),
            top_k,
        )

        # ---------------------------------------------------------
        # Generate scores
        # ---------------------------------------------------------

        scores = model.predict(
            pairs,
            show_progress_bar=False,
        )

        # ---------------------------------------------------------
        # Normalize scores into Python floats
        # ---------------------------------------------------------

        normalized_scores: list[float] = []

        for score in scores:
            try:
                normalized_scores.append(
                    float(score)
                )
            except (
                TypeError,
                ValueError,
            ):
                normalized_scores.append(
                    float("-inf")
                )

        # ---------------------------------------------------------
        # Attach scores to RetrievedDocument
        # ---------------------------------------------------------

        scored_documents: list[
            tuple[RetrievedDocument, float]
        ] = []

        for document, score in zip(
            valid_documents,
            normalized_scores,
        ):
            document.rerank_score = score

            scored_documents.append(
                (
                    document,
                    score,
                )
            )

        # ---------------------------------------------------------
        # Sort highest score first
        # ---------------------------------------------------------

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = [
            document
            for document, _ in scored_documents[:top_k]
        ]

        logger.info(
            "Reranking completed | returned={}",
            len(results),
        )

        logger.debug(
            "Reranker scores: {}",
            [
                round(
                    document.rerank_score,
                    4,
                )
                for document in results
            ],
        )

        return results

    def health_check(self) -> bool:
        """
        Verify that the reranker model can be loaded and used.
        """

        try:
            model = self._load_model()

            scores = model.predict(
                [
                    (
                        "machine learning",
                        "machine learning is a field of artificial intelligence",
                    )
                ],
                show_progress_bar=False,
            )

            if scores is None:
                return False

            logger.info(
                "Reranker health check passed."
            )

            return True

        except Exception:
            logger.exception(
                "Reranker health check failed."
            )
            return False