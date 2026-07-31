"""
Reciprocal Rank Fusion (RRF).

Combines multiple ranked retrieval lists into a single ranking.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.logging import logger
from app.rag.retrieval import RetrievedDocument


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedDocument],
    top_k: int = 10,
    k: int = 60,
) -> list[RetrievedDocument]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Score(document) = Σ 1 / (k + rank)
    """

    logger.info(
        "Running Reciprocal Rank Fusion on {} lists.",
        len(ranked_lists),
    )

    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, RetrievedDocument] = {}

    for ranked_list in ranked_lists:

        for rank, document in enumerate(
            ranked_list,
            start=1,
        ):

            key = document.content

            scores[key] += 1.0 / (k + rank)

            if key not in documents:
                documents[key] = document

    fused = sorted(
        documents.values(),
        key=lambda doc: scores[doc.content],
        reverse=True,
    )

    logger.info(
        "RRF produced {} unique documents.",
        len(fused),
    )

    return fused[:top_k]