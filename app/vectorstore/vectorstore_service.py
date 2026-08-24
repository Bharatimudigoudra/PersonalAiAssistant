"""
Vector Store Service.

High-level application service for vector database operations.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.vectorstore.factory import VectorStoreFactory


_vector_store: BaseVectorStore | None = None


class VectorStoreService:
    """
    Application-level vector store service.

    The underlying vector store is created only once.
    """

    def __init__(self) -> None:

        global _vector_store

        if _vector_store is None:

            logger.info(
                "Creating Vector Store singleton..."
            )

            _vector_store = (
                VectorStoreFactory.create()
            )

        self.store = _vector_store

        logger.info(
            "VectorStoreService initialized | provider={}",
            self.store.__class__.__name__,
        )

    # -----------------------------------------------------------------
    # Add
    # -----------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:

        self.store.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self,
        embedding: list[float],
        k: int = 5,
    ) -> dict[str, Any]:

        return self.store.search(
            embedding=embedding,
            k=k,
        )

    # -----------------------------------------------------------------
    # Count
    # -----------------------------------------------------------------

    def count(self) -> int:

        return self.store.count()

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health_check(self) -> bool:

        return self.store.health_check()


# ---------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------


_vectorstore_service: VectorStoreService | None = None


def get_vectorstore_service() -> VectorStoreService:
    """
    Return the application-level VectorStoreService singleton.
    """

    global _vectorstore_service

    if _vectorstore_service is None:

        logger.info(
            "Creating VectorStoreService singleton."
        )

        _vectorstore_service = (
            VectorStoreService()
        )

    return _vectorstore_service