"""
Document retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import rag, reranker
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.rag.reranker_service import RerankerService
from app.vectorstore.vectorstore_service import VectorStoreService

# ---------------------------------------------------------------------
# Singleton Services
# ---------------------------------------------------------------------

_embedding_service = EmbeddingService()
_vectorstore_service = VectorStoreService()


@dataclass(slots=True)
class RetrievedDocument:
    """
    Represents a retrieved document chunk.
    """

    content: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float = 0.0


class DocumentRetriever:
    """
    Retrieves relevant document chunks from the vector database.
    """

    def __init__(self) -> None:

        self.embedding_service = _embedding_service
        self.vectorstore = _vectorstore_service

        self.reranker = (
            RerankerService()
            if reranker.enabled
            else None
        )

        logger.info("DocumentRetriever initialized.")

    def retrieve(
        self,
        question: str,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve the most relevant document chunks.
        """

        start_time = time.perf_counter()

        if k is None:
            k = rag.top_k

        logger.info("Searching for: {}", question)
        logger.info("Top K: {}", k)

        query_embedding = self.embedding_service.embed_query(
            question,
        )

        results = self.vectorstore.search(
            embedding=query_embedding,
            k=k,
        )

        documents = results.get("documents") or [[]]
        distances = results.get("distances") or [[]]
        metadatas = results.get("metadatas") or [[]]

        if (
            not documents
            or not documents[0]
            or not distances
            or not distances[0]
        ):
            logger.warning(
                "Vector store returned no valid results."
            )
            return []

        logger.info(
            "Embedding search returned {} candidates.",
            len(documents[0]),
        )

        metadata_list = (
            metadatas[0]
            if metadatas and metadatas[0]
            else [{}] * len(documents[0])
        )

        retrieved: list[RetrievedDocument] = []
        seen: set[str] = set()

        for document, distance, metadata in zip(
            documents[0],
            distances[0],
            metadata_list,
        ):

            logger.debug(
                "Candidate distance: {:.4f}",
                distance,
            )

            if distance > rag.similarity_threshold:

                logger.debug(
                    "Rejected (distance {:.4f})",
                    distance,
                )

                continue

            if document in seen:
                continue

            seen.add(document)

            retrieved.append(
                RetrievedDocument(
                    content=document,
                    distance=distance,
                    metadata=metadata or {},
                )
            )

        logger.info(
            "{} documents remained after similarity filtering.",
            len(retrieved),
        )

        # Sort by embedding similarity
        retrieved.sort(
            key=lambda item: item.distance
        )

        # Cross Encoder Reranking
        if self.reranker is not None:

            logger.info(
                "Running CrossEncoder reranker..."
            )

            retrieved = self.reranker.rerank(
                query=question,
                documents=retrieved,
            )

            logger.info(
                "{} documents remained after reranking.",
                len(retrieved),
            )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Retrieval completed in {:.3f} seconds.",
            elapsed,
        )

        return retrieved