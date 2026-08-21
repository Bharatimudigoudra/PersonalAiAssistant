"""
Hybrid document retrieval for PersonalAiAssistant.

Pipeline:

    Interview Question
            |
            v
    Query Normalization
            |
            +----------------------+
            |                      |
            v                      v
      Dense Retrieval          BM25 Retrieval
            |                      |
            +----------+-----------+
                       |
                       v
              Candidate Merge
                       |
                       v
              Reciprocal Rank Fusion
                       |
                       v
                 Cross Encoder
                       |
                       v
                 Final Top-K

Design goals:
- Reliable retrieval
- No accidental loss of good candidates
- No duplicate chunks
- Persistent BM25 rebuild
- Cross-encoder reranking
- Clear logging
- Safe fallbacks
"""

from __future__ import annotations

import re
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
    Production-oriented hybrid document retriever.

    Retrieval strategy:

    1. Normalize the interview question.
    2. Optionally rewrite the query.
    3. Run dense vector retrieval.
    4. Run BM25 retrieval.
    5. Merge candidates with RRF.
    6. Rerank candidates using CrossEncoder.
    7. Deduplicate.
    8. Return final top-k documents.
    """

    # Number of candidates collected from each retrieval method
    # before fusion/reranking.
    CANDIDATE_MULTIPLIER = 3

    # Maximum number of candidates sent to the reranker.
    MAX_RERANK_CANDIDATES = 12

    def __init__(self) -> None:
        logger.info("Initializing DocumentRetriever...")

        # ---------------------------------------------------------
        # Core services
        # ---------------------------------------------------------

        self.embedding_service = EmbeddingService()

        self.vectorstore = VectorStoreService()

        self.bm25 = get_bm25_service()

        self.query_rewriter = get_query_rewriter_service()

        # ---------------------------------------------------------
        # Optional CrossEncoder
        # ---------------------------------------------------------

        self.reranker = (
            RerankerService()
            if reranker.enabled
            else None
        )

        # ---------------------------------------------------------
        # Rebuild BM25 from persistent ChromaDB
        # ---------------------------------------------------------

        self._ensure_bm25_index()

        logger.info("DocumentRetriever initialized successfully.")

    # =============================================================
    # PUBLIC API
    # =============================================================

    def retrieve(
        self,
        question: str,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for an interview question.

        Args:
            question:
                Interview question.

            k:
                Number of final documents to return.

        Returns:
            Ranked list of RetrievedDocument objects.
        """

        start = time.perf_counter()

        # ---------------------------------------------------------
        # Validate question
        # ---------------------------------------------------------

        question = self._normalize_question(question)

        if not question:
            logger.warning(
                "Empty interview question received."
            )
            return []

        # ---------------------------------------------------------
        # Determine final K
        # ---------------------------------------------------------

        final_k = self._resolve_k(k)

        # ---------------------------------------------------------
        # Candidate K
        #
        # We intentionally retrieve MORE candidates than the final
        # answer needs.
        #
        # Example:
        #
        # final_k = 3
        # candidate_k = 9
        #
        # This gives the reranker enough candidates to choose from.
        # ---------------------------------------------------------

        candidate_k = max(
            final_k * self.CANDIDATE_MULTIPLIER,
            8,
        )

        logger.info(
            "Starting retrieval | question={} | final_k={} | candidate_k={}",
            question,
            final_k,
            candidate_k,
        )

        # ---------------------------------------------------------
        # 1. Query rewriting
        # ---------------------------------------------------------

        query = self._rewrite_query(question)

        logger.info(
            "Retrieval query={}",
            query,
        )

        # ---------------------------------------------------------
        # 2. Dense vector retrieval
        # ---------------------------------------------------------

        vector_documents = self._vector_search(
            query=query,
            k=candidate_k,
        )

        logger.info(
            "Dense retrieval returned {} documents.",
            len(vector_documents),
        )

        # ---------------------------------------------------------
        # 3. BM25 retrieval
        # ---------------------------------------------------------

        bm25_documents = self._bm25_search(
            query=query,
            k=candidate_k,
        )

        logger.info(
            "BM25 retrieval returned {} documents.",
            len(bm25_documents),
        )

        # ---------------------------------------------------------
        # 4. If neither retriever found anything
        # ---------------------------------------------------------

        if not vector_documents and not bm25_documents:
            logger.warning(
                "No documents found by vector or BM25 retrieval."
            )

            self._log_elapsed(start)

            return []

        # ---------------------------------------------------------
        # 5. Deduplicate each retrieval list
        # ---------------------------------------------------------

        vector_documents = self._deduplicate_documents(
            vector_documents
        )

        bm25_documents = self._deduplicate_documents(
            bm25_documents
        )

        # ---------------------------------------------------------
        # 6. Reciprocal Rank Fusion
        #
        # IMPORTANT:
        # Do not prematurely reduce everything to final_k.
        # We want a larger candidate pool for reranking.
        # ---------------------------------------------------------

        fused_documents = self._fuse_results(
            vector_documents=vector_documents,
            bm25_documents=bm25_documents,
            candidate_k=candidate_k,
        )

        logger.info(
            "RRF produced {} candidate documents.",
            len(fused_documents),
        )

        if not fused_documents:
            self._log_elapsed(start)
            return []

        # ---------------------------------------------------------
        # 7. CrossEncoder reranking
        # ---------------------------------------------------------

        rerank_candidates = fused_documents[
            : self.MAX_RERANK_CANDIDATES
        ]

        if self.reranker is not None:
            rerank_candidates = self._rerank(
                query=query,
                documents=rerank_candidates,
            )

        # ---------------------------------------------------------
        # 8. Final deduplication
        # ---------------------------------------------------------

        final_documents = self._deduplicate_documents(
            rerank_candidates
        )

        # ---------------------------------------------------------
        # 9. Final top-k
        # ---------------------------------------------------------

        final_documents = final_documents[:final_k]

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------

        logger.info(
            "Final retrieval returned {} documents.",
            len(final_documents),
        )

        for index, document in enumerate(
            final_documents,
            start=1,
        ):
            preview = self._preview(
                document.content
            )

            source = (
                document.metadata.get("source")
                if document.metadata
                else None
            )

            logger.info(
                "Final document {} | source={} | preview={}",
                index,
                source or "unknown",
                preview,
            )

        self._log_elapsed(start)

        return final_documents

    # =============================================================
    # QUESTION NORMALIZATION
    # =============================================================

    @staticmethod
    def _normalize_question(
        question: str | None,
    ) -> str:
        """
        Normalize the interview question.

        This deliberately performs only light normalization.
        We do NOT rewrite the meaning of the question here.
        """

        if question is None:
            return ""

        question = str(question).strip()

        if not question:
            return ""

        # Collapse repeated whitespace.
        question = re.sub(
            r"\s+",
            " ",
            question,
        )

        return question

    # =============================================================
    # K RESOLUTION
    # =============================================================

    @staticmethod
    def _resolve_k(
        k: int | None,
    ) -> int:
        """
        Resolve final retrieval count.
        """

        if k is None:
            k = rag.reranker_top_k if hasattr(
                rag,
                "reranker_top_k",
            ) else reranker.top_k

        try:
            k = int(k)
        except (TypeError, ValueError):
            k = reranker.top_k

        if k <= 0:
            k = 1

        return k

    # =============================================================
    # QUERY REWRITING
    # =============================================================

    def _rewrite_query(
        self,
        question: str,
    ) -> str:
        """
        Rewrite the question when possible.

        If rewriting fails or produces an empty result,
        the original question is used.

        For simple interview questions such as:
            "Tell me about yourself?"
        the original question is already suitable.
        """

        # ---------------------------------------------------------
        # Avoid unnecessary rewriting for simple standalone
        # interview questions.
        # ---------------------------------------------------------

        simple_questions = {
            "tell me about yourself",
            "tell me about yourself?",
            "introduce yourself",
            "introduce yourself?",
            "what are your strengths",
            "what are your strengths?",
            "what are your weaknesses",
            "what are your weaknesses?",
        }

        normalized = question.lower().strip()

        if normalized in simple_questions:
            logger.info(
                "Simple interview question detected. "
                "Skipping query rewriting."
            )
            return question

        # ---------------------------------------------------------
        # Otherwise use query rewriter.
        # ---------------------------------------------------------

        try:
            rewritten = self.query_rewriter.rewrite(
                history="",
                question=question,
            )

            rewritten = (
                str(rewritten).strip()
                if rewritten
                else ""
            )

            if rewritten:
                return rewritten

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

        Chroma returns distances.

        IMPORTANT:
        Smaller distance means a closer/more similar vector.

        We therefore do not blindly interpret the value as a
        similarity score.

        The configured threshold is treated as an optional
        maximum-distance filter.
        """

        try:
            embedding = (
                self.embedding_service.embed_query(
                    query
                )
            )

            if not embedding:
                logger.warning(
                    "Query embedding is empty."
                )
                return []

            results = self.vectorstore.search(
                embedding=embedding,
                k=k,
            )

            return self._parse_vector_results(
                results,
                k=k,
            )

        except Exception:
            logger.exception(
                "Dense vector retrieval failed."
            )
            return []

    # =============================================================
    # VECTOR RESULT PARSER
    # =============================================================

    def _parse_vector_results(
        self,
        results: dict[str, Any] | None,
        k: int,
    ) -> list[RetrievedDocument]:
        """
        Convert Chroma result format into RetrievedDocument objects.
        """

        if not results:
            return []

        raw_documents = (
            results.get("documents")
            or []
        )

        raw_distances = (
            results.get("distances")
            or []
        )

        raw_metadatas = (
            results.get("metadatas")
            or []
        )

        if not raw_documents:
            return []

        documents = self._unwrap_chroma_list(
            raw_documents
        )

        distances = self._unwrap_chroma_list(
            raw_distances
        )

        metadatas = self._unwrap_chroma_list(
            raw_metadatas
        )

        logger.info(
            "Chroma returned {} raw candidates.",
            len(documents),
        )

        candidates: list[RetrievedDocument] = []

        for index, content in enumerate(
            documents
        ):
            if not content:
                continue

            content = str(content).strip()

            if not content:
                continue

            # -----------------------------------------------------
            # Distance
            # -----------------------------------------------------

            distance: float | None = None

            if index < len(distances):
                raw_distance = distances[index]

                if raw_distance is not None:
                    try:
                        distance = float(
                            raw_distance
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        distance = None

            # -----------------------------------------------------
            # Metadata
            # -----------------------------------------------------

            metadata: dict[str, Any] = {}

            if index < len(metadatas):
                raw_metadata = metadatas[index]

                if isinstance(
                    raw_metadata,
                    dict,
                ):
                    metadata = dict(
                        raw_metadata
                    )

            candidates.append(
                RetrievedDocument(
                    content=content,
                    distance=distance,
                    metadata=metadata,
                )
            )

            logger.debug(
                "Vector candidate {} | distance={} | source={}",
                index,
                (
                    f"{distance:.4f}"
                    if distance is not None
                    else "N/A"
                ),
                metadata.get("source", "unknown"),
            )

        # ---------------------------------------------------------
        # Sort by distance.
        # Smaller is better.
        # ---------------------------------------------------------

        candidates.sort(
            key=lambda document: (
                document.distance
                if document.distance is not None
                else float("inf")
            )
        )

        # ---------------------------------------------------------
        # Optional maximum-distance filter
        # ---------------------------------------------------------

        threshold = self._get_distance_threshold()

        if threshold is not None:
            filtered = [
                document
                for document in candidates
                if (
                    document.distance is None
                    or document.distance <= threshold
                )
            ]

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # Never destroy retrieval just because a threshold is
            # poorly tuned.
            #
            # If filtering removes everything, retain the closest
            # candidates.
            # -----------------------------------------------------

            if filtered:
                candidates = filtered
            else:
                logger.warning(
                    "Distance threshold {} rejected all "
                    "vector candidates. Keeping closest results.",
                    threshold,
                )

        candidates = self._deduplicate_documents(
            candidates
        )

        return candidates[:k]

    # =============================================================
    # CHROMA LIST NORMALIZATION
    # =============================================================

    @staticmethod
    def _unwrap_chroma_list(
        value: Any,
    ) -> list[Any]:
        """
        Normalize Chroma's nested list format.

        Example:

            [[doc1, doc2]]

        becomes:

            [doc1, doc2]
        """

        if not value:
            return []

        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], list)
        ):
            return value[0]

        if isinstance(value, list):
            return value

        return []

    # =============================================================
    # DISTANCE THRESHOLD
    # =============================================================

    @staticmethod
    def _get_distance_threshold() -> float | None:
        """
        Read the configured maximum vector distance.

        The project currently calls this setting
        `similarity_threshold`, but Chroma returns distance.

        Therefore:

            lower distance = better

        and the setting is interpreted as:

            distance <= threshold
        """

        value = getattr(
            rag,
            "similarity_threshold",
            None,
        )

        if value is None:
            return None

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            logger.warning(
                "Invalid similarity_threshold={}. "
                "Distance filtering disabled.",
                value,
            )
            return None

        if value <= 0:
            logger.warning(
                "similarity_threshold={} is not positive. "
                "Distance filtering disabled.",
                value,
            )
            return None

        return value

    # =============================================================
    # BM25 SEARCH
    # =============================================================

    def _bm25_search(
        self,
        query: str,
        k: int,
    ) -> list[RetrievedDocument]:
        """
        Run lexical BM25 retrieval.
        """

        try:
            documents = self.bm25.search(
                query=query,
                top_k=k,
            )

            return self._deduplicate_documents(
                documents or []
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
        Rebuild BM25 from the persistent Chroma collection.

        BM25 itself is in-memory, so this is necessary whenever
        the application starts in a new process.
        """

        try:
            collection = (
                self.vectorstore.store.collection
            )

            count = collection.count()

            logger.info(
                "Persistent Chroma collection contains {} documents.",
                count,
            )

            if count <= 0:
                logger.warning(
                    "Chroma collection is empty. "
                    "BM25 index will remain empty."
                )
                return

            result = collection.get(
                include=[
                    "documents",
                    "metadatas",
                ]
            )

            documents = (
                result.get("documents")
                or []
            )

            metadatas = (
                result.get("metadatas")
                or []
            )

            if not documents:
                logger.warning(
                    "Chroma collection returned no documents."
                )
                return

            bm25_documents: list[
                RetrievedDocument
            ] = []

            for index, content in enumerate(
                documents
            ):
                if not content:
                    continue

                content = str(
                    content
                ).strip()

                if not content:
                    continue

                metadata: dict[str, Any] = {}

                if index < len(metadatas):
                    raw_metadata = metadatas[index]

                    if isinstance(
                        raw_metadata,
                        dict,
                    ):
                        metadata = dict(
                            raw_metadata
                        )

                bm25_documents.append(
                    RetrievedDocument(
                        content=content,
                        distance=None,
                        metadata=metadata,
                    )
                )

            if not bm25_documents:
                logger.warning(
                    "No usable documents available for BM25."
                )
                return

            # Remove exact duplicate chunks before indexing.
            bm25_documents = (
                self._deduplicate_documents(
                    bm25_documents
                )
            )

            logger.info(
                "Building BM25 index with {} documents.",
                len(bm25_documents),
            )

            self.bm25.build_index(
                bm25_documents
            )

            logger.info(
                "BM25 index successfully rebuilt."
            )

        except Exception:
            logger.exception(
                "Failed to rebuild BM25 index from Chroma."
            )

    # =============================================================
    # RECIPROCAL RANK FUSION
    # =============================================================

    def _fuse_results(
        self,
        vector_documents: list[RetrievedDocument],
        bm25_documents: list[RetrievedDocument],
        candidate_k: int,
    ) -> list[RetrievedDocument]:
        """
        Fuse dense and lexical retrieval results.

        We keep a larger candidate pool here so that the
        CrossEncoder has enough candidates to make a useful
        ranking decision.
        """

        try:
            fused = reciprocal_rank_fusion(
                vector_documents,
                bm25_documents,
                top_k=candidate_k,
            )

            fused = self._deduplicate_documents(
                fused or []
            )

            return fused[:candidate_k]

        except Exception:
            logger.exception(
                "RRF failed. Using retrieval fallback."
            )

            fallback = (
                vector_documents
                if vector_documents
                else bm25_documents
            )

            return self._deduplicate_documents(
                fallback
            )[:candidate_k]

    # =============================================================
    # RERANKING
    # =============================================================

    def _rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank candidate documents using CrossEncoder.
        """

        if not documents:
            return []

        if self.reranker is None:
            return documents

        try:
            logger.info(
                "Reranking {} candidate documents.",
                len(documents),
            )

            reranked = self.reranker.rerank(
                query=query,
                documents=documents,
            )

            if not reranked:
                logger.warning(
                    "Reranker returned no documents. "
                    "Keeping original candidates."
                )
                return documents

            reranked = self._deduplicate_documents(
                reranked
            )

            logger.info(
                "Reranking completed. {} documents remain.",
                len(reranked),
            )

            return reranked

        except Exception:
            logger.exception(
                "CrossEncoder reranking failed. "
                "Keeping fused results."
            )

            return documents

    # =============================================================
    # DEDUPLICATION
    # =============================================================

    @staticmethod
    def _deduplicate_documents(
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Remove exact duplicate chunks.

        Priority:

        1. source + chunk metadata
        2. exact content

        This prevents duplicate Chroma entries from polluting
        the final context.
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

            source = str(
                metadata.get(
                    "source",
                    "",
                )
            ).strip()

            chunk = str(
                metadata.get(
                    "chunk",
                    "",
                )
            ).strip()

            # -----------------------------------------------------
            # Prefer source + chunk when available.
            # -----------------------------------------------------

            if source and chunk:
                document_id = (
                    f"{source}::{chunk}"
                )

                if document_id in seen_ids:
                    continue

                seen_ids.add(
                    document_id
                )

            # -----------------------------------------------------
            # Always protect against exact duplicate content.
            # -----------------------------------------------------

            content_key = content

            if content_key in seen_content:
                continue

            seen_content.add(
                content_key
            )

            unique.append(
                document
            )

        return unique

    # =============================================================
    # PREVIEW
    # =============================================================

    @staticmethod
    def _preview(
        content: str | None,
        limit: int = 180,
    ) -> str:
        """
        Create a short logging preview.
        """

        if not content:
            return ""

        text = re.sub(
            r"\s+",
            " ",
            str(content),
        ).strip()

        if len(text) <= limit:
            return text

        return (
            text[:limit]
            + "..."
        )

    # =============================================================
    # TIMING
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


# ================================================================
# Standalone retrieval test
# ================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PersonalAiAssistant - Retrieval Test")
    print("=" * 80)

    retriever = DocumentRetriever()

    while True:

        question = input(
            "\nInterview Question (exit to quit): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        results = retriever.retrieve(
            question,
            k=3,
        )

        print()
        print("=" * 80)
        print(
            f"RETRIEVED DOCUMENTS: {len(results)}"
        )
        print("=" * 80)

        if not results:
            print(
                "No relevant documents found."
            )
            continue

        for index, document in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"--- DOCUMENT {index} ---"
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
                "Distance:",
                document.distance,
            )

            print()
            print(document.content)

        print()
        print("=" * 80)