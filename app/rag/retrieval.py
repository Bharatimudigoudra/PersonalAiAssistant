"""
Hybrid document retrieval.

Pipeline:

    User Question
        ↓
    Query Rewriter
        ↓
    Dense Vector Search (Chroma)
        +
    BM25 Lexical Search
        ↓
    Reciprocal Rank Fusion
        ↓
    Cross-Encoder Reranker
        ↓
    Final Top-K Documents

Important:
    BM25 is an in-memory index. Therefore, when the application
    starts in a new Python process, the BM25 index must be rebuilt
    from the persistent ChromaDB collection.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import rag, reranker
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.models.retrieved_document import RetrievedDocument
from app.rag.bm25.bm25_service import get_bm25_service
from app.rag.query_rewriter.query_rewriter_service import (
    get_query_rewriter_service,
)
from app.rag.reciprocal_rank_fusion import reciprocal_rank_fusion
from app.rag.reranker_service import RerankerService
from app.vectorstore.vectorstore_service import VectorStoreService


class DocumentRetriever:
    """
    Hybrid document retriever.

    Combines:

    1. Dense vector retrieval
    2. BM25 lexical retrieval
    3. Reciprocal Rank Fusion
    4. Cross-Encoder reranking

    BM25 is automatically rebuilt from ChromaDB when the
    application starts.
    """

    def __init__(self) -> None:

        logger.info(
            "Initializing DocumentRetriever..."
        )

        # ---------------------------------------------------------
        # Services
        # ---------------------------------------------------------

        self.embedding_service = EmbeddingService()

        self.vectorstore = VectorStoreService()

        self.query_rewriter = (
            get_query_rewriter_service()
        )

        self.bm25 = get_bm25_service()

        # ---------------------------------------------------------
        # Optional reranker
        # ---------------------------------------------------------

        self.reranker = (
            RerankerService()
            if reranker.enabled
            else None
        )

        # ---------------------------------------------------------
        # Rebuild BM25 from persistent vector store
        # ---------------------------------------------------------

        self._ensure_bm25_index()

        logger.info(
            "DocumentRetriever initialized."
        )

    # =============================================================
    # PUBLIC API
    # =============================================================

    def retrieve(
        self,
        question: str,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve the most relevant documents.

        Args:
            question:
                User's interview question.

            k:
                Number of final documents to return.

        Returns:
            Ranked list of RetrievedDocument objects.
        """

        start = time.perf_counter()

        # ---------------------------------------------------------
        # Validate question
        # ---------------------------------------------------------

        if not question or not question.strip():

            logger.warning(
                "Empty retrieval question received."
            )

            return []

        question = question.strip()

        # ---------------------------------------------------------
        # Determine K
        # ---------------------------------------------------------

        if k is None:
            k = rag.top_k

        try:
            k = int(k)
        except (TypeError, ValueError):

            logger.warning(
                "Invalid retrieval k={}. Using configured top_k.",
                k,
            )

            k = rag.top_k

        if k <= 0:

            logger.warning(
                "Invalid retrieval k={}. Using k=1.",
                k,
            )

            k = 1

        logger.info(
            "Starting retrieval | question={} | k={}",
            question,
            k,
        )

        # ---------------------------------------------------------
        # 1. Query rewriting
        # ---------------------------------------------------------

        rewritten_question = self._rewrite_query(
            question
        )

        logger.info(
            "Standalone query={}",
            rewritten_question,
        )

        # ---------------------------------------------------------
        # 2. Dense vector retrieval
        # ---------------------------------------------------------

        vector_documents = self._vector_search(
            rewritten_question,
            k,
        )

        logger.info(
            "Vector search returned {} documents.",
            len(vector_documents),
        )

        # ---------------------------------------------------------
        # 3. BM25 retrieval
        # ---------------------------------------------------------

        bm25_documents = self._bm25_search(
            rewritten_question,
            k,
        )

        logger.info(
            "BM25 returned {} documents.",
            len(bm25_documents),
        )

        # ---------------------------------------------------------
        # 4. No results
        # ---------------------------------------------------------

        if not vector_documents and not bm25_documents:

            logger.warning(
                "No documents retrieved from vector search or BM25."
            )

            self._log_elapsed(start)

            return []

        # ---------------------------------------------------------
        # 5. Reciprocal Rank Fusion
        # ---------------------------------------------------------

        retrieved = self._fuse_results(
            vector_documents,
            bm25_documents,
            k,
        )

        logger.info(
            "RRF returned {} documents.",
            len(retrieved),
        )

        # ---------------------------------------------------------
        # 6. Cross-Encoder reranking
        # ---------------------------------------------------------

        if self.reranker is not None and retrieved:

            try:

                logger.info(
                    "Running CrossEncoder reranker..."
                )

                reranked = self.reranker.rerank(
                    query=rewritten_question,
                    documents=retrieved,
                )

                if reranked:

                    retrieved = reranked

                logger.info(
                    "Reranker returned {} documents.",
                    len(retrieved),
                )

            except Exception:

                logger.exception(
                    "Reranking failed. "
                    "Keeping RRF results."
                )

        # ---------------------------------------------------------
        # 7. Final cleanup
        # ---------------------------------------------------------

        retrieved = self._deduplicate_documents(
            retrieved
        )

        retrieved = retrieved[:k]

        self._log_elapsed(start)

        return retrieved

    # =============================================================
    # QUERY REWRITING
    # =============================================================

    def _rewrite_query(
        self,
        question: str,
    ) -> str:
        """
        Rewrite the user question into a standalone query.
        """

        try:

            rewritten = self.query_rewriter.rewrite(
                history="",
                question=question,
            )

            if rewritten and rewritten.strip():

                return rewritten.strip()

        except Exception:

            logger.exception(
                "Query rewriting failed. "
                "Using original question."
            )

        return question

    # =============================================================
    # VECTOR SEARCH
    # =============================================================

    def _vector_search(
        self,
        query: str,
        k: int,
    ) -> list[RetrievedDocument]:
        """
        Perform dense vector retrieval.

        Important:
            Chroma's distance is not necessarily cosine
            similarity. Smaller distance is better.

        If the configured threshold rejects every document,
        we fall back to the closest vector results instead
        of returning zero documents.
        """

        try:

            # -----------------------------------------------------
            # Generate query embedding
            # -----------------------------------------------------

            query_embedding = (
                self.embedding_service.embed_query(
                    query
                )
            )

            if not query_embedding:

                logger.warning(
                    "Embedding service returned an empty embedding."
                )

                return []

            # -----------------------------------------------------
            # Search Chroma
            # -----------------------------------------------------

            vector_results = self.vectorstore.search(
                embedding=query_embedding,
                k=max(k, 5),
            )

            return self._parse_vector_results(
                vector_results,
                k=k,
            )

        except Exception:

            logger.exception(
                "Vector retrieval failed."
            )

            return []

    # =============================================================
    # PARSE CHROMA RESULTS
    # =============================================================

    def _parse_vector_results(
        self,
        vector_results: dict[str, Any] | None,
        k: int,
    ) -> list[RetrievedDocument]:
        """
        Convert ChromaDB results into RetrievedDocument objects.

        Chroma usually returns:

            documents = [[doc1, doc2, ...]]

            distances = [[d1, d2, ...]]

            metadatas = [[meta1, meta2, ...]]

        We first collect all valid candidates.

        We then apply the configured distance threshold.

        If the threshold removes everything, we fall back to
        the closest candidates. This prevents a bad threshold
        from completely breaking retrieval.
        """

        if not vector_results:

            logger.warning(
                "Chroma returned an empty result."
            )

            return []

        raw_documents = (
            vector_results.get("documents")
            or []
        )

        raw_distances = (
            vector_results.get("distances")
            or []
        )

        raw_metadatas = (
            vector_results.get("metadatas")
            or []
        )

        # ---------------------------------------------------------
        # Validate nested Chroma structure
        # ---------------------------------------------------------

        if not raw_documents:

            logger.warning(
                "Chroma response contains no documents."
            )

            return []

        documents_list = (
            raw_documents[0]
            if isinstance(raw_documents[0], list)
            else raw_documents
        )

        distances_list = (
            raw_distances[0]
            if raw_distances
            and isinstance(raw_distances[0], list)
            else []
        )

        metadata_list = (
            raw_metadatas[0]
            if raw_metadatas
            and isinstance(raw_metadatas[0], list)
            else []
        )

        logger.info(
            "Raw Chroma returned {} documents.",
            len(documents_list),
        )

        # ---------------------------------------------------------
        # Build candidates
        # ---------------------------------------------------------

        candidates: list[
            RetrievedDocument
        ] = []

        for index, document in enumerate(
            documents_list
        ):

            if not document:
                continue

            # -----------------------------------------------------
            # Distance
            # -----------------------------------------------------

            distance: float | None = None

            if (
                index < len(distances_list)
                and distances_list[index] is not None
            ):

                try:

                    distance = float(
                        distances_list[index]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    logger.warning(
                        "Invalid Chroma distance at index={}.",
                        index,
                    )

            # -----------------------------------------------------
            # Metadata
            # -----------------------------------------------------

            metadata: dict[str, Any] = {}

            if (
                index < len(metadata_list)
                and metadata_list[index]
            ):

                raw_metadata = (
                    metadata_list[index]
                )

                if isinstance(
                    raw_metadata,
                    dict,
                ):

                    metadata = raw_metadata

            logger.info(
                "Vector candidate | index={} | distance={}",
                index,
                (
                    f"{distance:.4f}"
                    if distance is not None
                    else "N/A"
                ),
            )

            candidates.append(
                RetrievedDocument(
                    content=document,
                    distance=distance,
                    metadata=metadata,
                )
            )

        if not candidates:

            logger.warning(
                "No valid Chroma candidates found."
            )

            return []

        # ---------------------------------------------------------
        # Sort by distance
        #
        # Smaller distance = more similar.
        # ---------------------------------------------------------

        candidates.sort(
            key=lambda document: (
                document.distance
                if document.distance is not None
                else float("inf")
            )
        )

        # ---------------------------------------------------------
        # Apply configured threshold
        # ---------------------------------------------------------

        threshold = getattr(
            rag,
            "similarity_threshold",
            None,
        )

        accepted: list[
            RetrievedDocument
        ] = []

        if threshold is not None:

            try:

                threshold = float(
                    threshold
                )

            except (
                TypeError,
                ValueError,
            ):

                logger.warning(
                    "Invalid similarity threshold={}. "
                    "Threshold filtering disabled.",
                    threshold,
                )

                threshold = None

        if threshold is not None:

            for document in candidates:

                if (
                    document.distance is None
                    or document.distance <= threshold
                ):

                    accepted.append(
                        document
                    )

        else:

            accepted = list(
                candidates
            )

        # ---------------------------------------------------------
        # CRITICAL FALLBACK
        #
        # If threshold is too strict and rejects everything,
        # keep the closest candidates.
        #
        # Your Chroma tests showed distances around 1.05-1.23,
        # so a threshold such as 0.5 can reject every result.
        # ---------------------------------------------------------

        if not accepted:

            logger.warning(
                "Similarity threshold rejected all "
                "vector candidates. "
                "Falling back to closest documents."
            )

            accepted = candidates

        # ---------------------------------------------------------
        # Deduplicate
        # ---------------------------------------------------------

        accepted = self._deduplicate_documents(
            accepted
        )

        accepted = accepted[:k]

        logger.info(
            "Vector search accepted {} documents.",
            len(accepted),
        )

        return accepted

    # =============================================================
    # BM25 SEARCH
    # =============================================================

    def _bm25_search(
        self,
        query: str,
        k: int,
    ) -> list[RetrievedDocument]:
        """
        Perform BM25 retrieval.

        The BM25 index is rebuilt from ChromaDB during
        DocumentRetriever initialization, so this works
        even when the interview assistant is started
        as a separate Python process.
        """

        try:

            return self.bm25.search(
                query=query,
                top_k=k,
            )

        except Exception:

            logger.exception(
                "BM25 retrieval failed."
            )

            return []

    # =============================================================
    # BM25 INITIALIZATION
    # =============================================================

    def _ensure_bm25_index(self) -> None:
        """
        Rebuild BM25 index from all persistent Chroma documents.
        """

        try:
            collection = self.vectorstore.store.collection

            count = collection.count()

            logger.info(
                "Persistent vector collection contains %d documents.",
                count,
            )

            if count == 0:
                logger.warning(
                    "Vector collection is empty. "
                    "BM25 index will remain empty."
                )
                return

            result = collection.get(
                include=[
                    "documents",
                    "metadatas",
                ]
            )

            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []

            if not documents:
                logger.warning(
                    "No documents found in persistent vector collection."
                )
                return

            retrieved_documents: list[RetrievedDocument] = []

            for index, content in enumerate(documents):

                if not content:
                    continue

                metadata = (
                    metadatas[index]
                    if index < len(metadatas)
                    and metadatas[index]
                    else {}
                )

                retrieved_documents.append(
                    RetrievedDocument(
                        content=content,
                        distance=None,
                        metadata=metadata,
                    )
                )

            logger.info(
                "Rebuilding BM25 index from %d persistent documents...",
                len(retrieved_documents),
            )

            self.bm25.build_index(
                retrieved_documents,
            )

            logger.info(
                "BM25 index successfully rebuilt with %d documents.",
                len(retrieved_documents),
            )

        except Exception:
            logger.exception(
                "Failed to rebuild BM25 index."
            )

    # =============================================================
    # RRF
    # =============================================================

    def _fuse_results(
        self,
        vector_documents: list[RetrievedDocument],
        bm25_documents: list[RetrievedDocument],
        k: int,
    ) -> list[RetrievedDocument]:
        """
        Combine vector and BM25 results using RRF.
        """

        try:

            retrieved = reciprocal_rank_fusion(
                vector_documents,
                bm25_documents,
                top_k=k,
            )

            return (
                self._deduplicate_documents(
                    retrieved
                )[:k]
            )

        except Exception:

            logger.exception(
                "Reciprocal Rank Fusion failed."
            )

            # -----------------------------------------------------
            # Safe fallback
            # -----------------------------------------------------

            fallback = (
                vector_documents
                if vector_documents
                else bm25_documents
            )

            return (
                self._deduplicate_documents(
                    fallback
                )[:k]
            )

    # =============================================================
    # DEDUPLICATION
    # =============================================================

    @staticmethod
    def _deduplicate_documents(
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Remove duplicate retrieved documents.

        Deduplication is based primarily on content.
        """

        unique: list[
            RetrievedDocument
        ] = []

        seen: set[str] = set()

        for document in documents:

            if not document:
                continue

            content = (
                document.content
                or ""
            ).strip()

            if not content:
                continue

            key = content

            if key in seen:
                continue

            seen.add(key)

            unique.append(
                document
            )

        return unique

    # =============================================================
    # LOGGING
    # =============================================================

    @staticmethod
    def _log_elapsed(
        start: float,
    ) -> None:
        """
        Log retrieval execution time.
        """

        elapsed = (
            time.perf_counter()
            - start
        )

        logger.info(
            "Retrieval completed in {:.3f} sec.",
            elapsed,
        )