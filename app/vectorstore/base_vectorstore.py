"""
Base Vector Store Interface.

Defines the contract that every vector-store implementation
must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """
    Abstract interface for vector databases.
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add or update documents in the vector store.

        Implementations should safely handle existing IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Search the vector store using an embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Return the number of stored documents/chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check whether the vector store is healthy.
        """
        raise NotImplementedError