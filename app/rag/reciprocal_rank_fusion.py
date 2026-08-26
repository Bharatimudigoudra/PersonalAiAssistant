"""
Reciprocal Rank Fusion (RRF).

Combines multiple ranked retrieval lists into a single ranking.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedDocument],
    top_k: int = 10,
    k: int = 60,
) -> list[RetrievedDocument]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF score:

        score(d) = sum(1 / (k + rank))

    Higher scores are better.
    """

    if top_k <= 0:
        return []

    if k < 1:
        k = 60

    logger.info(
        "Running Reciprocal Rank Fusion | lists={}",
        len(ranked_lists),
    )

    scores: dict[str, float] = defaultdict(float)

    documents: dict[
        str,
        RetrievedDocument,
    ] = {}

    for ranked_list in ranked_lists:

        if not ranked_list:
            continue

        for rank, document in enumerate(
            ranked_list,
            start=1,
        ):

            if document is None:
                continue

            content = str(
                document.content or ""
            ).strip()

            if not content:
                continue

            # Content is used as the stable deduplication key.
            key = content

            scores[key] += 1.0 / (
                k + rank
            )

            if key not in documents:
                documents[key] = document

    fused = sorted(
        documents.values(),
        key=lambda document: scores[
            str(document.content or "").strip()
        ],
        reverse=True,
    )

    logger.info(
        "RRF produced {} unique documents.",
        len(fused),
    )

    return fused[:top_k]