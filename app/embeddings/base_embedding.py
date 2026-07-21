"""
Abstract embedding interface.
"""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """
    Base class for all embedding providers.
    """

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Embed a single query.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        """
        Embed multiple documents.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check whether the embedding model is available.
        """
        raise NotImplementedError