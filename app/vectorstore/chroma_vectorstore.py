"""
ChromaDB vector store implementation.
"""

from chromadb import PersistentClient

from app.core.config import vectorstore
from app.core.logging import logger
from app.vectorstore.base_vectorstore import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation.
    """

    def __init__(self) -> None:

        logger.info("Initializing ChromaDB...")

        self.client = PersistentClient(
            path=vectorstore.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=vectorstore.collection_name
        )

        logger.info(
            "Collection loaded: {}",
            vectorstore.collection_name,
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Store embeddings.
        """

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Inserted {} documents.",
            len(ids),
        )

    def search(
        self,
        embedding: list[float],
        k: int = 5,
    ):

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
        )

    def health_check(self) -> bool:

        try:

            self.client.heartbeat()

            logger.info(
                "ChromaDB healthy."
            )

            return True

        except Exception as exc:

            logger.exception(exc)

            return False