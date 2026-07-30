"""
Embedding Service.

Provides a singleton interface for generating embeddings.
"""

from app.core.logging import logger
from app.embeddings.base_embedding import BaseEmbedding
from app.embeddings.factory import EmbeddingFactory


# Singleton embedding provider
_embedding_provider: BaseEmbedding | None = None


class EmbeddingService:
    """
    High-level embedding service.

    The embedding model is loaded only once and shared across
    the entire application.
    """

    def __init__(self) -> None:

        global _embedding_provider

        if _embedding_provider is None:

            logger.info(
                "Creating Embedding Model singleton..."
            )

            _embedding_provider = EmbeddingFactory.create()

        self.provider = _embedding_provider

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a query.
        """

        return self.provider.embed_query(text)

    def embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """

        return self.provider.embed_documents(documents)

    def health_check(
        self,
    ) -> bool:
        """
        Verify embedding model availability.
        """

        return self.provider.health_check()