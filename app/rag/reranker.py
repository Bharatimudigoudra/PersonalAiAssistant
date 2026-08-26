"""
Cross-Encoder document reranker.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import reranker
from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument


class DocumentReranker:
    """
    Reranks retrieved documents using a Sentence Transformers
    CrossEncoder model.
    """

    def __init__(self) -> None:

        logger.info(
            "Loading reranker model: {}",
            reranker.model_name,
        )

        self.model = CrossEncoder(
            reranker.model_name,
        )

        logger.info(
            "Reranker loaded successfully | model={}",
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

        if not query or not query.strip():
            logger.warning(
                "Cannot rerank with an empty query."
            )
            return documents

        if not documents:
            logger.warning(
                "No documents available for reranking."
            )
            return []

        query = query.strip()

        logger.info(
            "Reranking {} documents...",
            len(documents),
        )

        pairs = [
            (
                query,
                document.content,
            )
            for document in documents
        ]

        try:
            raw_scores = self.model.predict(
                pairs,
                show_progress_bar=False,
            )

        except Exception:
            logger.exception(
                "CrossEncoder prediction failed."
            )
            raise

        scores = self._normalize_scores(
            raw_scores
        )

        if len(scores) != len(documents):
            raise RuntimeError(
                "CrossEncoder returned an unexpected "
                "number of scores: "
                f"{len(scores)} for {len(documents)} documents."
            )

        ranked: list[
            tuple[RetrievedDocument, float]
        ] = []

        for document, score in zip(
            documents,
            scores,
        ):
            document.rerank_score = score

            ranked.append(
                (
                    document,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        top_k = max(
            1,
            int(reranker.top_k),
        )

        ranked = ranked[:top_k]

        logger.info(
            "Reranking completed | input={} | output={}",
            len(documents),
            len(ranked),
        )

        logger.debug(
            "Top reranker scores: {}",
            [
                round(score, 4)
                for _, score in ranked
            ],
        )

        return [
            document
            for document, _ in ranked
        ]

    @staticmethod
    def _normalize_scores(
        scores: Any,
    ) -> list[float]:
        """
        Convert CrossEncoder output into a normal Python
        list of floats.
        """

        if scores is None:
            return []

        # NumPy ndarray / tensor-like objects.
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        if not isinstance(scores, (list, tuple)):
            scores = [scores]

        normalized: list[float] = []

        for score in scores:

            # Handle unexpected nested single-value lists.
            if isinstance(
                score,
                (list, tuple),
            ):
                if not score:
                    continue

                score = score[0]

            try:
                normalized.append(
                    float(score)
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise RuntimeError(
                    f"Invalid CrossEncoder score: {score!r}"
                ) from exc

        return normalized