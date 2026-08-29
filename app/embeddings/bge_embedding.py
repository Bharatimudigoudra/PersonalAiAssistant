"""
BGE Embedding Provider.

Uses SentenceTransformers with the configured BGE model.

The same model and normalization configuration MUST be used
for both:
    - document embeddings during ingestion
    - query embeddings during retrieval
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.core.config import embedding
from app.core.logging import logger
from app.embeddings.base_embedding import BaseEmbedding


class BGEEmbedding(BaseEmbedding):
    """
    BGE embedding implementation using SentenceTransformer.
    """

    def __init__(self) -> None:

        model_name = embedding.model_name
        device = embedding.device

        logger.info(
            "Loading embedding model: {}",
            model_name,
        )

        logger.info(
            "Embedding device: {}",
            device,
        )

        logger.info(
            "Normalize embeddings: {}",
            embedding.normalize_embeddings,
        )

        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )

        except Exception:
            logger.exception(
                "Failed to load embedding model: {}",
                model_name,
            )

            raise

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = (
            embedding.normalize_embeddings
        )

        # Determine embedding dimensionality once.
        # Newer sentence-transformers versions renamed this API;
        # keep compatibility with both names.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

        if self.dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        logger.info(
            "Embedding model loaded successfully | "
            "model={} | dimension={}",
            self.model_name,
            self.dimension,
        )

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a search query.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Query must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Query cannot be empty."
            )

        try:
            vector = self.model.encode(
                text,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        except Exception:
            logger.exception(
                "Failed to generate query embedding."
            )

            raise

        result = vector.tolist()

        if len(result) != self.dimension:
            raise RuntimeError(
                "Query embedding dimension mismatch: "
                f"expected={self.dimension}, "
                f"actual={len(result)}"
            )

        return result

    # ------------------------------------------------------------------
    # Document embeddings
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """

        if not isinstance(documents, list):
            raise TypeError(
                "Documents must be a list of strings."
            )

        if not documents:
            return []

        cleaned_documents: list[str] = []

        for index, document in enumerate(documents):

            if not isinstance(document, str):
                raise TypeError(
                    f"Document {index} must be a string."
                )

            document = document.strip()

            if not document:
                raise ValueError(
                    f"Document {index} cannot be empty."
                )

            cleaned_documents.append(document)

        try:
            vectors = self.model.encode(
                cleaned_documents,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        except Exception:
            logger.exception(
                "Failed to generate document embeddings."
            )

            raise

        result = vectors.tolist()

        if len(result) != len(cleaned_documents):
            raise RuntimeError(
                "Document embedding count mismatch: "
                f"expected={len(cleaned_documents)}, "
                f"actual={len(result)}"
            )

        for index, vector in enumerate(result):

            if len(vector) != self.dimension:
                raise RuntimeError(
                    "Document embedding dimension mismatch: "
                    f"document={index}, "
                    f"expected={self.dimension}, "
                    f"actual={len(vector)}"
                )

        return result

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Verify that the embedding model can generate vectors.
        """

        try:

            vector = self.embed_query(
                "embedding health check"
            )

            valid = (
                len(vector) == self.dimension
                and len(vector) > 0
            )

            if valid:
                logger.info(
                    "Embedding health check passed | dimension={}",
                    self.dimension,
                )

            else:
                logger.error(
                    "Embedding health check failed."
                )

            return valid

        except Exception:
            logger.exception(
                "Embedding health check failed."
            )

            return False