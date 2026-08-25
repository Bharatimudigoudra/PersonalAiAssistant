"""
Document retrieval service.

Initial retrieval pipeline:

    Question
       ↓
    Embedding
       ↓
    ChromaDB
       ↓
    RetrievedDocument
       ↓
    Deduplication
       ↓
    Optional distance filtering
       ↓
    Top-K
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.core.config import rag
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.models.retrieved_document import RetrievedDocument
from app.vectorstore.vectorstore_service import VectorStoreService


class DocumentRetriever:
    """
    Dense vector document retriever.

    BM25, RRF and CrossEncoder reranking should be added only
    after dense retrieval has been verified independently.
    """

    def __init__(self) -> None:
        logger.info("Initializing DocumentRetriever...")

        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStoreService()

        logger.info(
            "DocumentRetriever initialized successfully."
        )

    # ==========================================================
    # PUBLIC
    # ==========================================================

    def retrieve(
        self,
        question: str,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve the most relevant document chunks.
        """

        start = time.perf_counter()

        question = self._normalize_question(question)

        if not question:
            logger.warning(
                "Empty retrieval question."
            )
            return []

        final_k = self._resolve_k(k)

        logger.info(
            "Retrieval started | question=%s | k=%s",
            question,
            final_k,
        )

        # ------------------------------------------------------
        # Create query embedding
        # ------------------------------------------------------

        try:
            query_embedding = (
                self.embedding_service.embed_query(
                    question
                )
            )
        except Exception:
            logger.exception(
                "Failed to generate query embedding."
            )
            return []

        if not query_embedding:
            logger.warning(
                "Query embedding is empty."
            )
            return []

        logger.info(
            "Query embedding generated | dimensions=%s",
            len(query_embedding),
        )

        # ------------------------------------------------------
        # Chroma search
        # ------------------------------------------------------

        try:
            search_result = self.vectorstore.search(
                embedding=query_embedding,
                k=final_k,
            )
        except Exception:
            logger.exception(
                "Chroma vector search failed."
            )
            return []

        # ------------------------------------------------------
        # Parse Chroma result
        # ------------------------------------------------------

        documents = self._parse_chroma_results(
            search_result
        )

        logger.info(
            "Chroma returned %s usable documents.",
            len(documents),
        )

        # ------------------------------------------------------
        # Deduplicate
        # ------------------------------------------------------

        documents = self._deduplicate(
            documents
        )

        # ------------------------------------------------------
        # Sort by distance
        # ------------------------------------------------------

        documents.sort(
            key=lambda document: (
                document.distance
                if document.distance is not None
                else float("inf")
            )
        )

        # ------------------------------------------------------
        # Optional distance threshold
        # ------------------------------------------------------

        threshold = (
            rag.similarity_threshold
        )

        if threshold is not None:
            filtered = [
                document
                for document in documents
                if (
                    document.distance is not None
                    and document.distance <= threshold
                )
            ]

            logger.info(
                "Distance filtering | threshold=%s | "
                "before=%s | after=%s",
                threshold,
                len(documents),
                len(filtered),
            )

            # Do not accidentally return zero results because
            # of a badly tuned threshold.
            if filtered:
                documents = filtered

        # ------------------------------------------------------
        # Final K
        # ------------------------------------------------------

        documents = documents[:final_k]

        # ------------------------------------------------------
        # Diagnostic logging
        # ------------------------------------------------------

        for index, document in enumerate(
            documents,
            start=1,
        ):
            logger.info(
                "RESULT %s | distance=%s | source=%s | chunk=%s",
                index,
                document.distance,
                document.metadata.get(
                    "source",
                    "unknown",
                ),
                document.metadata.get(
                    "chunk",
                    "unknown",
                ),
            )

        elapsed = (
            time.perf_counter() - start
        )

        logger.info(
            "Retrieval completed in %.3f seconds.",
            elapsed,
        )

        return documents

    # ==========================================================
    # QUESTION NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_question(
        question: str | None,
    ) -> str:
        """
        Lightly normalize a question.
        """

        if question is None:
            return ""

        question = str(
            question
        ).strip()

        if not question:
            return ""

        return re.sub(
            r"\s+",
            " ",
            question,
        )

    # ==========================================================
    # K
    # ==========================================================

    @staticmethod
    def _resolve_k(
        k: int | None,
    ) -> int:
        """
        Resolve the requested number of results.
        """

        if k is None:
            k = rag.top_k

        try:
            k = int(k)
        except (
            TypeError,
            ValueError,
        ):
            k = rag.top_k

        return max(
            1,
            k,
        )

    # ==========================================================
    # CHROMA PARSER
    # ==========================================================

    @classmethod
    def _parse_chroma_results(
        cls,
        results: dict[str, Any] | None,
    ) -> list[RetrievedDocument]:
        """
        Convert Chroma's nested response into
        RetrievedDocument objects.
        """

        if not results:
            return []

        ids = cls._unwrap(
            results.get("ids")
        )

        documents = cls._unwrap(
            results.get("documents")
        )

        distances = cls._unwrap(
            results.get("distances")
        )

        metadatas = cls._unwrap(
            results.get("metadatas")
        )

        retrieved: list[
            RetrievedDocument
        ] = []

        for index, content in enumerate(
            documents
        ):
            if content is None:
                continue

            content = str(
                content
            ).strip()

            if not content:
                continue

            # --------------------------------------------------
            # Distance
            # --------------------------------------------------

            distance = None

            if index < len(distances):
                raw_distance = (
                    distances[index]
                )

                try:
                    if raw_distance is not None:
                        distance = float(
                            raw_distance
                        )
                except (
                    TypeError,
                    ValueError,
                ):
                    distance = None

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------

            metadata: dict[str, Any] = {}

            if index < len(metadatas):
                raw_metadata = (
                    metadatas[index]
                )

                if isinstance(
                    raw_metadata,
                    dict,
                ):
                    metadata = dict(
                        raw_metadata
                    )

            # Keep Chroma ID available.
            if index < len(ids):
                metadata.setdefault(
                    "id",
                    ids[index],
                )

            retrieved.append(
                RetrievedDocument(
                    content=content,
                    distance=distance,
                    metadata=metadata,
                )
            )

        return retrieved

    # ==========================================================
    # CHROMA NORMALIZATION
    # ==========================================================

    @staticmethod
    def _unwrap(
        value: Any,
    ) -> list[Any]:
        """
        Chroma normally returns:

            [[item1, item2, item3]]

        Convert it to:

            [item1, item2, item3]
        """

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            return []

        if (
            len(value) == 1
            and isinstance(
                value[0],
                list,
            )
        ):
            return value[0]

        return value

    # ==========================================================
    # DEDUPLICATION
    # ==========================================================

    @staticmethod
    def _deduplicate(
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Remove duplicate chunks.
        """

        unique: list[
            RetrievedDocument
        ] = []

        seen_ids: set[str] = set()
        seen_content: set[str] = set()

        for document in documents:

            if document is None:
                continue

            content = str(
                document.content or ""
            ).strip()

            if not content:
                continue

            metadata = (
                document.metadata
                if isinstance(
                    document.metadata,
                    dict,
                )
                else {}
            )

            document_id = str(
                metadata.get(
                    "id",
                    "",
                )
            ).strip()

            if document_id:
                if document_id in seen_ids:
                    continue

                seen_ids.add(
                    document_id
                )

            content_key = content.lower()

            if content_key in seen_content:
                continue

            seen_content.add(
                content_key
            )

            unique.append(
                document
            )

        return unique


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PersonalAiAssistant - Dense Retrieval Test")
    print("=" * 80)

    retriever = DocumentRetriever()

    while True:

        question = input(
            "\nInterview Question "
            "(type 'exit' to quit): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        results = retriever.retrieve(
            question,
            k=8,
        )

        print()
        print("=" * 80)
        print(
            f"RESULTS: {len(results)}"
        )
        print("=" * 80)

        if not results:
            print(
                "No documents retrieved."
            )
            continue

        for index, document in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"DOCUMENT {index}"
            )
            print("-" * 80)

            print(
                "Distance:",
                document.distance,
            )

            print(
                "Source:",
                document.metadata.get(
                    "source",
                    "unknown",
                ),
            )

            print(
                "Chunk:",
                document.metadata.get(
                    "chunk",
                    "unknown",
                ),
            )

            print(
                "ID:",
                document.metadata.get(
                    "id",
                    "unknown",
                ),
            )

            print()
            print(
                document.content
            )

    print()
    print("=" * 80)
    print("Retrieval test finished.")
    print("=" * 80)