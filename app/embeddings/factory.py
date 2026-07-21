"""
Embedding factory.
"""

from app.embeddings.bge_embedding import BGEEmbedding
from app.embeddings.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """
    Creates embedding providers.
    """

    @staticmethod
    def create() -> BaseEmbedding:
        """
        Return the configured embedding provider.
        """

        return BGEEmbedding()