"""
Hybrid document retrieval.
"""

from __future__ import annotations

import time

from app.models.retrieved_document import RetrievedDocument
from app.core.config import rag, reranker
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.rag.bm25.bm25_service import (
    get_bm25_service,
)
from app.rag.query_rewriter.query_rewriter_service import (
    get_query_rewriter_service,
)
from app.rag.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
)
from app.rag.reranker_service import RerankerService
from app.vectorstore.vectorstore_service import (
    VectorStoreService,
)

_embedding_service = EmbeddingService()
_vectorstore_service = VectorStoreService()
_query_rewriter = get_query_rewriter_service()
_bm25_service = get_bm25_service()


class DocumentRetriever:
    """
    Hybrid Retriever.
    """

    def __init__(self) -> None:

        self.embedding_service = _embedding_service
        self.vectorstore = _vectorstore_service
        self.query_rewriter = _query_rewriter
        self.bm25 = _bm25_service

        self.reranker = (
            RerankerService()
            if reranker.enabled
            else None
        )

        logger.info(
            "DocumentRetriever initialized."
        )

    def retrieve(
        self,
        question: str,
        k: int | None = None,
    ) -> list[RetrievedDocument]:

        start = time.perf_counter()

        if k is None:
            k = rag.top_k

        logger.info(
            "Original query: {}",
            question,
        )

        rewritten_question = self.query_rewriter.rewrite(
            history="",
            question=question,
        )

        logger.info(
            "Standalone query: {}",
            rewritten_question,
        )

        query_embedding = (
            self.embedding_service.embed_query(
                rewritten_question,
            )
        )

        vector_results = self.vectorstore.search(
            embedding=query_embedding,
            k=k,
        )

        vector_documents: list[RetrievedDocument] = []

        docs = vector_results.get("documents") or [[]]
        dists = vector_results.get("distances") or [[]]
        metas = vector_results.get("metadatas") or [[]]

        if docs and docs[0]:

            metadata_list = (
                metas[0]
                if metas and metas[0]
                else [{}] * len(docs[0])
            )

            for document, distance, metadata in zip(
                docs[0],
                dists[0],
                metadata_list,
            ):

                if distance > rag.similarity_threshold:
                    continue

                vector_documents.append(
                    RetrievedDocument(
                        content=document,
                        distance=distance,
                        metadata=metadata or {},
                    )
                )

        logger.info(
            "Vector search returned {} documents.",
            len(vector_documents),
        )

        bm25_documents = self.bm25.search(
            query=rewritten_question,
            top_k=k,
        )

        logger.info(
            "BM25 returned {} documents.",
            len(bm25_documents),
        )

        retrieved = reciprocal_rank_fusion(
            vector_documents,
            bm25_documents,
            top_k=k,
        )

        logger.info(
            "RRF produced {} documents.",
            len(retrieved),
        )

        if (
            self.reranker is not None
            and retrieved
        ):

            logger.info(
                "Running CrossEncoder..."
            )

            retrieved = self.reranker.rerank(
                query=rewritten_question,
                documents=retrieved,
            )

        elapsed = time.perf_counter() - start

        logger.info(
            "Retrieval completed in {:.3f} sec.",
            elapsed,
        )

        return retrieved