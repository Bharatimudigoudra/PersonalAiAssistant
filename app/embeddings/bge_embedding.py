"""
BGE Embedding Provider.

Uses Sentence Transformers to generate embeddings.
"""

from sentence_transformers import SentenceTransformer

from app.core.config import embedding
from app.core.logging import logger
from app.embeddings.base_embedding import BaseEmbedding


class BGEEmbedding(BaseEmbedding):
    """
    HuggingFace BGE embedding implementation.
    """

    def __init__(self) -> None:
        logger.info(
            "Loading embedding model: {}",
            embedding.model_name,
        )

        self.model = SentenceTransformer(
            embedding.model_name,
            device=embedding.device,
        )

        logger.info("Embedding model loaded successfully.")

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate embedding for a single query.
        """

        vector = self.model.encode(
            text,
            normalize_embeddings=embedding.normalize_embeddings,
        )

        return vector.tolist()

    def embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """

        vectors = self.model.encode(
            documents,
            normalize_embeddings=embedding.normalize_embeddings,
        )

        return vectors.tolist()

    def health_check(self) -> bool:
        """
        Check whether the embedding model is working.
        """

        try:
            self.embed_query("health check")

            logger.info("Embedding health check passed.")

            return True

        except Exception:
            logger.exception("Embedding health check failed.")

            return False