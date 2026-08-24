"""
Vector Store Factory.

Creates the vector store configured in config.yaml.
"""

from __future__ import annotations

from app.core.config import vectorstore
from app.core.logging import logger
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.vectorstore.chroma_vectorstore import ChromaVectorStore


class VectorStoreFactory:
    """
    Factory for vector store implementations.
    """

    @staticmethod
    def create() -> BaseVectorStore:

        provider = (
            vectorstore.provider
            .strip()
            .lower()
        )

        logger.info(
            "Selected Vector Store: {}",
            provider,
        )

        if provider == "chroma":
            return ChromaVectorStore()

        raise ValueError(
            "Unsupported vector store provider: "
            f"{vectorstore.provider}"
        )