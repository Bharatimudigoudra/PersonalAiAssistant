"""
Cross Encoder reranker.
"""

from sentence_transformers import CrossEncoder

from app.core.config import reranker
from app.core.logging import logger
from app.rag.retrieval import RetrievedDocument


class DocumentReranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
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
            "Reranker loaded successfully."
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank retrieved documents based on semantic relevance.
        """

        if not documents:

            logger.warning(
                "No documents available for reranking."
            )

            return []

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

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        logger.info(
            "Returning top {} reranked documents.",
            reranker.top_k,
        )

        logger.debug(
            "Top reranker scores: {}",
            [
                round(score, 4)
                for _, score in ranked[:reranker.top_k]
            ],
        )

        return [
            document
            for document, _ in ranked[:reranker.top_k]
        ]