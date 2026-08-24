"""
ChromaDB vector store implementation.

Uses:
    - Persistent ChromaDB storage
    - cosine distance
    - upsert semantics
    - validated embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from app.core.config import BASE_DIR, vectorstore
from app.core.logging import logger
from app.vectorstore.base_vectorstore import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation for persistent vector storage.
    """

    def __init__(self) -> None:
        logger.info("Initializing ChromaDB...")

        # -------------------------------------------------------------
        # Resolve persistent path relative to project root
        # -------------------------------------------------------------

        persist_path = Path(
            vectorstore.persist_directory
        )

        if not persist_path.is_absolute():
            persist_path = BASE_DIR / persist_path

        persist_path = persist_path.resolve()

        persist_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.persist_directory = persist_path

        logger.info(
            "ChromaDB persistence path: {}",
            self.persist_directory,
        )

        # -------------------------------------------------------------
        # Create persistent client
        # -------------------------------------------------------------

        self.client = PersistentClient(
            path=str(self.persist_directory)
        )

        # -------------------------------------------------------------
        # Create / load collection
        #
        # IMPORTANT:
        # cosine distance is explicitly selected because the
        # embedding service uses normalized embeddings.
        # -------------------------------------------------------------

        self.collection = (
            self.client.get_or_create_collection(
                name=vectorstore.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                },
            )
        )

        logger.info(
            "ChromaDB collection loaded | name={} | count={}",
            vectorstore.collection_name,
            self.collection.count(),
        )

    # -----------------------------------------------------------------
    # Add / Update
    # -----------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Insert or update document chunks.

        Uses upsert instead of add so re-ingesting the same
        document IDs does not create duplicate-ID errors.
        """

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        if not ids:
            raise ValueError(
                "Cannot add documents: ids list is empty."
            )

        if not documents:
            raise ValueError(
                "Cannot add documents: documents list is empty."
            )

        if not embeddings:
            raise ValueError(
                "Cannot add documents: embeddings list is empty."
            )

        if not metadatas:
            raise ValueError(
                "Cannot add documents: metadatas list is empty."
            )

        if not (
            len(ids)
            == len(documents)
            == len(embeddings)
            == len(metadatas)
        ):
            raise ValueError(
                "Vector store input lengths do not match: "
                f"ids={len(ids)}, "
                f"documents={len(documents)}, "
                f"embeddings={len(embeddings)}, "
                f"metadatas={len(metadatas)}"
            )

        # -------------------------------------------------------------
        # Validate embedding dimensions
        # -------------------------------------------------------------

        embedding_dimensions = {
            len(vector)
            for vector in embeddings
        }

        if len(embedding_dimensions) != 1:
            raise ValueError(
                "Embedding vectors have inconsistent dimensions: "
                f"{embedding_dimensions}"
            )

        dimension = next(
            iter(embedding_dimensions)
        )

        if dimension <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        # -------------------------------------------------------------
        # Validate text
        # -------------------------------------------------------------

        cleaned_documents: list[str] = []

        for index, document in enumerate(documents):

            text = str(
                document or ""
            ).strip()

            if not text:
                raise ValueError(
                    f"Document at index {index} is empty."
                )

            cleaned_documents.append(text)

        # -------------------------------------------------------------
        # Normalize metadata
        # -------------------------------------------------------------

        cleaned_metadatas: list[dict[str, Any]] = []

        for metadata in metadatas:

            if metadata is None:
                cleaned_metadatas.append({})

            else:
                cleaned_metadatas.append(
                    dict(metadata)
                )

        # -------------------------------------------------------------
        # Upsert
        # -------------------------------------------------------------

        logger.info(
            "Upserting {} documents into ChromaDB | dimension={}",
            len(ids),
            dimension,
        )

        self.collection.upsert(
            ids=ids,
            documents=cleaned_documents,
            embeddings=embeddings,
            metadatas=cleaned_metadatas,
        )

        logger.info(
            "ChromaDB upsert completed | total_documents={}",
            self.collection.count(),
        )

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self,
        embedding: list[float],
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Search ChromaDB using cosine distance.
        """

        if not embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "Search k must be greater than zero."
            )

        collection_count = self.collection.count()

        if collection_count == 0:
            logger.warning(
                "Vector search requested but collection is empty."
            )

            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        # Never request more results than exist.
        actual_k = min(
            k,
            collection_count,
        )

        logger.info(
            "Running ChromaDB vector search | requested_k={} | actual_k={} | collection_count={}",
            k,
            actual_k,
            collection_count,
        )

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=actual_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        # -------------------------------------------------------------
        # Logging search results
        # -------------------------------------------------------------

        distances = (
            results.get("distances") or [[]]
        )

        if distances and distances[0]:

            for index, distance in enumerate(
                distances[0]
            ):
                logger.info(
                    "Vector result | rank={} | cosine_distance={:.4f}",
                    index + 1,
                    float(distance),
                )

        logger.info(
            "ChromaDB search completed | results={}",
            len(distances[0]) if distances else 0,
        )

        return results

    # -----------------------------------------------------------------
    # Count
    # -----------------------------------------------------------------

    def count(self) -> int:
        """
        Return number of stored chunks.
        """

        return int(
            self.collection.count()
        )

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Check ChromaDB availability.
        """

        try:

            self.client.heartbeat()

            count = self.collection.count()

            logger.info(
                "ChromaDB health check passed | documents={}",
                count,
            )

            return True

        except Exception:

            logger.exception(
                "ChromaDB health check failed."
            )

            return False