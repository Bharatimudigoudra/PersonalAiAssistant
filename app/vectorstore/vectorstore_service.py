"""
Vector Store Service.

Provides a singleton interface to the configured vector store.
"""

from app.core.logging import logger
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.vectorstore.factory import VectorStoreFactory


# Singleton vector store instance
_vector_store: BaseVectorStore | None = None


class VectorStoreService:
    """
    High-level service for interacting with the configured
    vector database.

    The underlying vector store is created only once and
    shared across the entire application.
    """

    def __init__(self) -> None:

        global _vector_store

        if _vector_store is None:

            logger.info(
                "Creating Vector Store singleton..."
            )

            _vector_store = VectorStoreFactory.create()

        self.store = _vector_store

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Store documents and embeddings.
        """

        self.store.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        embedding: list[float],
        k: int = 5,
    ):
        """
        Search the vector database.
        """

        return self.store.search(
            embedding=embedding,
            k=k,
        )

    def health_check(
        self,
    ) -> bool:
        """
        Verify vector store availability.
        """

        return self.store.health_check()