"""
Embedding Service.

Application-level interface for generating embeddings.

Responsibilities:
- Create the embedding provider once.
- Reuse the same provider across the application.
- Validate embedding inputs.
- Provide query and document embedding methods.
- Expose a health check.
"""

from __future__ import annotations

from app.core.logging import logger
from app.embeddings.base_embedding import BaseEmbedding
from app.embeddings.factory import EmbeddingFactory


_embedding_provider: BaseEmbedding | None = None


class EmbeddingService:
    """
    High-level embedding service.

    The underlying embedding model is created once and reused.
    """

    def __init__(self) -> None:
        global _embedding_provider

        if _embedding_provider is None:
            logger.info(
                "Creating Embedding Model singleton."
            )

            _embedding_provider = EmbeddingFactory.create()

            logger.info(
                "Embedding provider created: {}",
                _embedding_provider.__class__.__name__,
            )

        self.provider = _embedding_provider

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate one embedding for a search query.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Embedding query must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Embedding query cannot be empty."
            )

        logger.debug(
            "Generating query embedding | chars={}",
            len(text),
        )

        vector = self.provider.embed_query(text)

        if not vector:
            raise RuntimeError(
                "Embedding provider returned an empty query vector."
            )

        return vector

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
                "Documents must be provided as a list."
            )

        if not documents:
            return []

        cleaned_documents: list[str] = []

        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    f"Document at index {index} must be a string."
                )

            document = document.strip()

            if not document:
                raise ValueError(
                    f"Document at index {index} is empty."
                )

            cleaned_documents.append(document)

        logger.debug(
            "Generating document embeddings | count={}",
            len(cleaned_documents),
        )

        vectors = self.provider.embed_documents(
            cleaned_documents
        )

        if len(vectors) != len(cleaned_documents):
            raise RuntimeError(
                "Embedding provider returned an unexpected "
                "number of document vectors."
            )

        for index, vector in enumerate(vectors):
            if not vector:
                raise RuntimeError(
                    f"Empty embedding returned for document {index}."
                )

        return vectors

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Verify that the embedding provider is functional.
        """

        try:
            result = self.provider.health_check()

            if result:
                logger.info(
                    "Embedding service health check: PASS."
                )
            else:
                logger.error(
                    "Embedding service health check: FAIL."
                )

            return result

        except Exception:
            logger.exception(
                "Embedding service health check failed."
            )

            return False


# ----------------------------------------------------------------------
# Singleton accessor
# ----------------------------------------------------------------------

_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Return the application-level embedding service singleton.
    """

    global _service

    if _service is None:
        logger.info(
            "Creating EmbeddingService singleton."
        )

        _service = EmbeddingService()

    return _service


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":

    service = get_embedding_service()

    print("=" * 70)
    print("Embedding Service Test")
    print("=" * 70)

    query_vector = service.embed_query(
        "Tell me about yourself."
    )

    print(
        f"Query embedding dimensions: {len(query_vector)}"
    )

    document_vectors = service.embed_documents(
        [
            "Bharati is a Data Scientist.",
            "Bharati has experience in Generative AI and Machine Learning.",
        ]
    )

    print(
        f"Document count: {len(document_vectors)}"
    )

    print(
        f"Document dimensions: {len(document_vectors[0])}"
    )

    print(
        f"Health check: {service.health_check()}"
    )

    print("=" * 70)
    print("Embedding Service Test Completed")
    print("=" * 70)