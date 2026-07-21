"""
Document retrieval.
"""

from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vectorstore_service import VectorStoreService


class DocumentRetriever:
    """
    Retrieves relevant document chunks from the vector database.
    """

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStoreService()

    def retrieve(
        self,
        question: str,
        k: int = 3,
    ) -> list[str]:
        """
        Retrieve the most relevant document chunks.
        """

        logger.info("Searching for: {}", question)

        query_embedding = self.embedding_service.embed_query(question)

        results = self.vectorstore.search(
            query_embedding,
            k,
        )

        documents = results.get("documents", [[]])[0]

        # Remove duplicate chunks while preserving order
        documents = list(dict.fromkeys(documents))

        logger.info(
            "Retrieved {} unique chunks.",
            len(documents),
        )

        return documents