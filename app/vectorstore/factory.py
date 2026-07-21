"""
Vector store factory.
"""

from app.core.config import vectorstore
from app.core.logging import logger
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.vectorstore.chroma_vectorstore import ChromaVectorStore


class VectorStoreFactory:

    @staticmethod
    def create() -> BaseVectorStore:

        provider = vectorstore.provider.lower()

        logger.info(
            "Selected Vector Store: {}",
            provider,
        )

        if provider == "chroma":
            return ChromaVectorStore()

        raise ValueError(
            f"Unsupported vector store: {provider}"
        )