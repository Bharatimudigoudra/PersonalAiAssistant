"""
BM25 retrieval implementation.
"""

from rank_bm25 import BM25Okapi

from app.core.logging import logger
from app.rag.retrieval import RetrievedDocument


class BM25Retriever:
    """
    BM25 keyword retriever.
    """

    def __init__(self) -> None:

        self._bm25 = None
        self._documents: list[RetrievedDocument] = []

        logger.info(
            "BM25Retriever initialized."
        )

    def build_index(
        self,
        documents: list[RetrievedDocument],
    ) -> None:
        """
        Build a BM25 index.
        """

        self._documents = documents

        corpus = [
            document.content.lower().split()
            for document in documents
        ]

        self._bm25 = BM25Okapi(
            corpus,
        )

        logger.info(
            "BM25 index built with {} documents.",
            len(documents),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        """
        Search the BM25 index.
        """

        if (
            self._bm25 is None
            or not self._documents
        ):

            logger.warning(
                "BM25 index is empty."
            )

            return []

        logger.info(
            "Running BM25 search..."
        )

        tokens = query.lower().split()

        scores = self._bm25.get_scores(
            tokens,
        )

        ranked = sorted(
            zip(
                self._documents,
                scores,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        results = [
            document
            for document, score in ranked
            if score > 0
        ]

        logger.info(
            "BM25 returned {} documents.",
            len(results[:top_k]),
        )

        return results[:top_k]