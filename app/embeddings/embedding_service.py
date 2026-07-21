"""
Embedding service.
"""

from app.embeddings.factory import EmbeddingFactory


class EmbeddingService:
    """
    High-level embedding service.
    """

    def __init__(self):

        self.provider = EmbeddingFactory.create()

    def embed_query(
        self,
        text: str,
    ):

        return self.provider.embed_query(text)

    def embed_documents(
        self,
        documents: list[str],
    ):

        return self.provider.embed_documents(documents)

    def health_check(self):

        return self.provider.health_check()