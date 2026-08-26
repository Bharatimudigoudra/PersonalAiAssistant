"""
CrossEncoder document reranker.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import reranker
from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument


class DocumentReranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """

    def __init__(self) -> None:

        if not reranker.enabled:
            raise RuntimeError(
                "Reranker is disabled in configuration."
            )

        logger.info(
            "Loading reranker model: {}",
            reranker.model_name,
        )

        self.model = CrossEncoder(
            reranker.model_name,
        )

        logger.info(
            "Reranker model loaded successfully | model={}",
            reranker.model_name,
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rank documents according to query-document relevance.
        """

        if not documents:
            logger.warning(
                "No documents available for reranking."
            )
            return []

        query = str(query or "").strip()

        if not query:
            logger.warning(
                "Cannot rerank with an empty query."
            )
            return documents

        logger.info(
            "Reranking {} documents...",
            len(documents),
        )

        pairs = [
            (
                query,
                str(document.content or ""),
            )
            for document in documents
        ]

        try:

            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
            )

        except Exception:
            logger.exception(
                "CrossEncoder prediction failed."
            )
            return documents

        ranked: list[
            tuple[RetrievedDocument, float]
        ] = []

        for document, score in zip(
            documents,
            scores,
        ):

            try:
                score_value = float(score)
            except (
                TypeError,
                ValueError,
            ):
                score_value = float("-inf")

            # Store score on the document.
            #
            # This works even if RetrievedDocument does not
            # explicitly define rerank_score, but we should
            # update the model later to make it type-safe.
            try:
                document.rerank_score = score_value
            except Exception:
                pass

            ranked.append(
                (
                    document,
                    score_value,
                )
            )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        top_k = self._resolve_top_k()

        final_results = [
            document
            for document, _ in ranked[:top_k]
        ]

        logger.info(
            "Reranking completed | candidates={} | returned={}",
            len(documents),
            len(final_results),
        )

        for index, (
            document,
            score,
        ) in enumerate(
            ranked[:top_k],
            start=1,
        ):

            metadata = (
                document.metadata
                if isinstance(
                    document.metadata,
                    dict,
                )
                else {}
            )

            logger.info(
                "RERANK RESULT {} | score={:.4f} | distance={} | source={} | chunk={}",
                index,
                score,
                document.distance,
                metadata.get(
                    "source",
                    "unknown",
                ),
                metadata.get(
                    "chunk",
                    "unknown",
                ),
            )

        return final_results

    @staticmethod
    def _resolve_top_k() -> int:
        """
        Safely resolve reranker top-k.
        """

        try:
            value = int(
                reranker.top_k
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 3

        return max(
            value,
            1,
        )