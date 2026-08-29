"""
Document Retrieval Service.

Pipeline:

    Question
       |
       v
    Query normalization
       |
       v
    Dense vector retrieval
       |
       v
    Candidate filtering
       |
       v
    CrossEncoder reranking
       |
       v
    Deduplication
       |
       v
    Final top-k documents

The first goal is reliable semantic retrieval.
BM25/RRF can be added after this pipeline is verified.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.core.config import rag, reranker
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.models.retrieved_document import RetrievedDocument
from app.rag.reranker_service import RerankerService
from app.vectorstore.vectorstore_service import VectorStoreService


class DocumentRetriever:
    """
    Retrieves relevant interview-context documents.

    Current pipeline:

        Question
            ↓
        BGE embedding
            ↓
        ChromaDB
            ↓
        Candidate documents
            ↓
        CrossEncoder reranker
            ↓
        Deduplication
            ↓
        Final top-k
    """

    # Retrieve more candidates than we finally need.
    CANDIDATE_MULTIPLIER = 4

    # Never send an unnecessarily large number of chunks
    # to the CrossEncoder.
    MAX_RERANK_CANDIDATES = 20

    def __init__(self) -> None:
        logger.info("Initializing DocumentRetriever...")

        # Defer heavy dependencies until the first query is actually
        # executed. This avoids downloading/loading embedding and rerank
        # models just by constructing the retriever.
        self.embedding_service: EmbeddingService | None = None
        self.vectorstore: VectorStoreService | None = None
        self.reranker: RerankerService | None = None

        # --------------------------------------------------------
        # Retrieval configuration
        # --------------------------------------------------------

        self.candidate_multiplier = getattr(
            rag,
            "candidate_multiplier",
            4,
        )

        self.max_rerank_candidates = getattr(
            rag,
            "max_rerank_candidates",
            16,
        )

        logger.info(
            "DocumentRetriever initialized | "
            "reranker_enabled={} | "
            "candidate_multiplier={} | "
            "max_rerank_candidates={}",
            reranker.enabled,
            self.candidate_multiplier,
            self.max_rerank_candidates,
        )

    def _ensure_dependencies(self) -> None:
        """
        Lazily create heavy retrieval dependencies on first use.
        """

        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()

        if self.vectorstore is None:
            self.vectorstore = VectorStoreService()

        if self.reranker is None and reranker.enabled:
            self.reranker = RerankerService()

    # ============================================================
    # PUBLIC API
    # ============================================================

    def retrieve(
        self,
        question: str,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve the most relevant document chunks.

        Args:
            question: Interview question.
            k: Number of final documents.

        Returns:
            Ranked list of RetrievedDocument objects.
        """

        start = time.perf_counter()

        question = self._normalize_question(question)

        if not question:
            logger.warning(
                "Empty retrieval question."
            )
            return []

        self._ensure_dependencies()

        final_k = self._resolve_k(k)

        candidate_k = max(
            final_k * self.candidate_multiplier,
            8,
        )

        logger.info(
            "Retrieval started | question={} | final_k={} | candidate_k={}",
            question,
            final_k,
            candidate_k,
        )

        # --------------------------------------------------------
        # 1. Generate query embedding
        # --------------------------------------------------------

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
            "Query embedding generated | dimensions={}",
            len(query_embedding),
        )

        # --------------------------------------------------------
        # 2. Dense vector search
        # --------------------------------------------------------

        try:
            search_result = self.vectorstore.search(
                embedding=query_embedding,
                k=candidate_k,
            )
        except Exception:
            logger.exception(
                "Vector store search failed."
            )
            return []

        candidates = self._parse_vector_results(
            search_result
        )

        logger.info(
            "Dense retrieval returned {} candidates.",
            len(candidates),
        )

        if not candidates:
            logger.warning(
                "No documents retrieved from vector store."
            )
            return []

        # --------------------------------------------------------
        # 3. Log candidates BEFORE reranking
        # --------------------------------------------------------

        self._log_candidates(
            candidates,
            title="DENSE RETRIEVAL RESULTS",
        )

        # --------------------------------------------------------
        # 4. Reranking
        # --------------------------------------------------------

        rerank_candidates = candidates[
            : self.max_rerank_candidates
        ]

        if self.reranker is not None:
            reranked = self._rerank(
                question,
                rerank_candidates,
            )

            if reranked:
                candidates = reranked

        candidates = self._boost_by_lexical_relevance(
            question,
            candidates,
        )
        candidates = self._boost_by_source_relevance(
            question,
            candidates,
        )

        # --------------------------------------------------------
        # 5. Deduplicate
        # --------------------------------------------------------

        candidates = self._deduplicate_documents(
            candidates
        )

        # --------------------------------------------------------
        # 6. Final top-k
        # --------------------------------------------------------

        final_documents = candidates[:final_k]

        self._log_candidates(
            final_documents,
            title="FINAL RETRIEVAL RESULTS",
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "Retrieval completed | results={} | elapsed={:.3f}s",
            len(final_documents),
            elapsed,
        )

        return final_documents

    # ============================================================
    # QUESTION NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_question(
        question: str | None,
    ) -> str:
        """
        Perform light normalization only.

        We intentionally do not rewrite the question here.
        """

        if question is None:
            return ""

        question = str(question).strip()

        if not question:
            return ""

        question = re.sub(
            r"\s+",
            " ",
            question,
        )

        return question

    # ============================================================
    # K
    # ============================================================

    @staticmethod
    def _resolve_k(
        k: int | None,
    ) -> int:
        """
        Resolve final number of documents.
        """

        if k is None:
            k = reranker.top_k

        try:
            k = int(k)
        except (TypeError, ValueError):
            k = reranker.top_k

        if k <= 0:
            k = 1

        return k

    # ============================================================
    # PARSE CHROMA RESULTS
    # ============================================================

    def _parse_vector_results(
        self,
        results: dict[str, Any] | None,
    ) -> list[RetrievedDocument]:
        """
        Convert Chroma's nested result format into
        RetrievedDocument objects.
        """

        if not results:
            return []

        documents = self._unwrap(
            results.get("documents")
        )

        distances = self._unwrap(
            results.get("distances")
        )

        metadatas = self._unwrap(
            results.get("metadatas")
        )

        ids = self._unwrap(
            results.get("ids")
        )

        if not documents:
            return []

        candidates: list[RetrievedDocument] = []

        for index, raw_content in enumerate(
            documents
        ):
            content = str(
                raw_content or ""
            ).strip()

            if not content:
                continue

            # ----------------------------------------------------
            # Distance
            # ----------------------------------------------------

            distance = float("inf")

            if index < len(distances):
                try:
                    distance = float(
                        distances[index]
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    distance = float("inf")

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

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

            # Keep Chroma ID available for diagnostics.
            if index < len(ids):
                metadata.setdefault(
                    "_vector_id",
                    str(ids[index]),
                )

            candidates.append(
                RetrievedDocument(
                    content=content,
                    distance=distance,
                    metadata=metadata,
                )
            )

        # --------------------------------------------------------
        # Chroma cosine distance:
        #
        # smaller = more similar
        # --------------------------------------------------------

        candidates.sort(
            key=lambda document: document.distance
        )

        return self._deduplicate_documents(
            candidates
        )

    # ============================================================
    # CHROMA NORMALIZATION
    # ============================================================

    @staticmethod
    def _unwrap(
        value: Any,
    ) -> list[Any]:
        """
        Convert:

            [[a, b, c]]

        into:

            [a, b, c]
        """

        if value is None:
            return []

        if isinstance(value, list):
            if (
                len(value) == 1
                and isinstance(value[0], list)
            ):
                return value[0]

            return value

        return []

    # ============================================================
    # RERANKING
    # ============================================================

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
            logger.info(
                "Reranker disabled. Keeping fused results."
            )
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

            for index, document in enumerate(
                reranked,
                start=1,
            ):
                logger.info(
                    "RERANKED RESULT {} | score={} | distance={} | source={} | chunk={}",
                    index,
                    (
                        f"{document.rerank_score:.4f}"
                        if document.rerank_score is not None
                        else "None"
                    ),
                    (
                        f"{document.distance:.4f}"
                        if document.distance is not None
                        else "None"
                    ),
                    document.metadata.get(
                        "source",
                        "unknown",
                    ),
                    document.metadata.get(
                        "chunk",
                        "unknown",
                    ),
                )

            return reranked

        except Exception:
            logger.exception(
                "CrossEncoder reranking failed. "
                "Keeping fused results."
            )

            return documents
    def _boost_by_lexical_relevance(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Add a lightweight lexical relevance bonus so experience and
        project-oriented questions rank their matching evidence above
        generic intros or unrelated project chunks.
        """

        if not documents:
            return []

        question = self._normalize_question(query).lower()

        if not question:
            return documents

        question_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                question,
            )
        )

        def lexical_score(document: RetrievedDocument) -> float:
            text = re.sub(
                r"\s+",
                " ",
                str(document.content or "").lower(),
            ).strip()

            if not text:
                return 0.0

            bonus = 0.0

            experience_query = any(
                term in question_tokens
                for term in {
                    "internship",
                    "intern",
                    "experience",
                    "role",
                }
            )

            if "yourself" in question_tokens and not experience_query:
                if "introduce yourself" in text or "myself" in text:
                    bonus += 2.5

            if experience_query:
                explicit_experience_weights = {
                    "work experience": 5.0,
                    "ai associate": 4.5,
                    "resourcepro": 4.5,
                    "internship": 4.0,
                    "intern": 4.0,
                    "role": 3.5,
                    "experience": 2.0,
                    "developed": 1.5,
                    "worked": 1.5,
                    "responsibilities": 1.5,
                }

                for phrase, weight in explicit_experience_weights.items():
                    if phrase in text:
                        bonus += weight
                        break

            if "python" in question_tokens and "python" in text:
                bonus += 2.0

            if "project" in question_tokens and (
                "project" in text or "developed" in text
            ):
                bonus += 1.5

            overlap = sum(
                1
                for token in question_tokens
                if token and token in text
            )
            bonus += 0.2 * overlap

            rerank_score = getattr(
                document,
                "rerank_score",
                None,
            )
            if rerank_score is None:
                rerank_score = 0.0

            return bonus + float(rerank_score)

        return sorted(
            documents,
            key=lambda document: (
                lexical_score(document),
                float(
                    getattr(
                        document,
                        "rerank_score",
                        0.0,
                    )
                    or 0.0
                ),
                -(
                    float(document.distance)
                    if document.distance is not None
                    else 0.0
                ),
            ),
            reverse=True,
        )

    def _boost_by_source_relevance(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Prefer the document type that most naturally matches the question:
        resume/pdf for work experience and internship questions, and
        interview/pdf for self-introduction questions.
        """

        if not documents:
            return []

        question = self._normalize_question(query).lower()
        if not question:
            return documents

        experience_query = any(
            term in question
            for term in {
                "experience",
                "internship",
                "intern",
                "role",
                "work experience",
                "professional experience",
            }
        )

        intro_query = "yourself" in question or "introduce" in question

        def source_priority(document: RetrievedDocument) -> float:
            metadata = (
                document.metadata
                if isinstance(document.metadata, dict)
                else {}
            )
            source = str(metadata.get("source", "")).lower()

            if not source:
                return 0.0

            if experience_query and "resume" in source:
                return 5.0
            if intro_query and "interview" in source:
                return 5.0

            if experience_query and "interview" in source:
                return -2.0
            if intro_query and "resume" in source:
                return -2.0

            return 0.0

        return sorted(
            documents,
            key=lambda document: (
                source_priority(document),
                float(
                    getattr(document, "rerank_score", 0.0)
                    or 0.0
                ),
                -(
                    float(document.distance)
                    if document.distance is not None
                    else 0.0
                ),
            ),
            reverse=True,
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate_documents(
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Remove duplicate chunks.

        Priority:

        1. Exact vector ID
        2. source + chunk
        3. Exact content
        """

        unique: list[RetrievedDocument] = []

        seen_ids: set[str] = set()
        seen_source_chunks: set[str] = set()
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

            # ----------------------------------------------------
            # Chroma vector ID
            # ----------------------------------------------------

            vector_id = str(
                metadata.get(
                    "_vector_id",
                    "",
                )
            ).strip()

            if vector_id:
                if vector_id in seen_ids:
                    continue

                seen_ids.add(vector_id)

            # ----------------------------------------------------
            # Source + chunk
            # ----------------------------------------------------

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

            if source and chunk:
                source_chunk = (
                    f"{source}::{chunk}"
                )

                if (
                    source_chunk
                    in seen_source_chunks
                ):
                    continue

                seen_source_chunks.add(
                    source_chunk
                )

            # ----------------------------------------------------
            # Exact content
            # ----------------------------------------------------

            content_key = re.sub(
                r"\s+",
                " ",
                content,
            ).strip().lower()

            if content_key in seen_content:
                continue

            seen_content.add(
                content_key
            )

            unique.append(
                document
            )

        return unique

    # ============================================================
    # LOGGING
    # ============================================================

    @staticmethod
    def _log_candidates(
        documents: list[RetrievedDocument],
        title: str,
    ) -> None:
        """
        Print useful retrieval diagnostics.
        """

        logger.info(
            "{} | count={}",
            title,
            len(documents),
        )

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = (
                document.metadata
                if isinstance(
                    document.metadata,
                    dict,
                )
                else {}
            )

            source = metadata.get(
                "source",
                "unknown",
            )

            chunk = metadata.get(
                "chunk",
                "unknown",
            )

            distance = document.distance

            rerank_score = getattr(
                document,
                "rerank_score",
                None,
            )

            preview = re.sub(
                r"\s+",
                " ",
                document.content,
            ).strip()

            if len(preview) > 180:
                preview = (
                    preview[:180]
                    + "..."
                )

            logger.info(
                "{} | rank={} | distance={} | "
                "rerank_score={} | source={} | chunk={} | preview={}",
                title,
                index,
                (
                    f"{distance:.4f}"
                    if distance != float("inf")
                    else "N/A"
                ),
                rerank_score,
                source,
                chunk,
                preview,
            )


# ================================================================
# Standalone test
# ================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PersonalAiAssistant - Retrieval Test")
    print("=" * 80)

    retriever = DocumentRetriever()

    while True:

        try:
            question = input(
                "\nInterview Question "
                "(type 'exit' to quit): "
            ).strip()

        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if question.lower() in {
            "exit",
            "quit",
        }:
            print(
                "\nRetrieval test finished."
            )
            break

        if not question:
            print(
                "\nPlease enter an interview question."
            )
            continue

        results = retriever.retrieve(
            question,
            k=reranker.top_k,
        )

        print()
        print("=" * 80)
        print(
            f"FINAL RESULTS: {len(results)}"
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
            metadata = (
                document.metadata
                if isinstance(
                    document.metadata,
                    dict,
                )
                else {}
            )

            print()
            print(
                f"DOCUMENT {index}"
            )
            print("-" * 80)

            print(
                "Source   :",
                metadata.get(
                    "source",
                    "unknown",
                ),
            )

            print(
                "Chunk    :",
                metadata.get(
                    "chunk",
                    "unknown",
                ),
            )

            print(
                "Distance :",
                document.distance,
            )

            print(
                "Rerank   :",
                getattr(
                    document,
                    "rerank_score",
                    None,
                ),
            )

            print()
            print(
                document.content
            )

    print("=" * 80)