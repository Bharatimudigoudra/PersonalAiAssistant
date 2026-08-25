"""
Reciprocal Rank Fusion.

Combines multiple ranked retrieval lists into a single ranking.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument


def _document_key(
    document: RetrievedDocument,
) -> str:
    """
    Generate a stable deduplication key.

    Prefer source + chunk when available.
    Fall back to document content.
    """

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

    if source and chunk:
        return f"{source}::{chunk}"

    return str(
        document.content or ""
    ).strip()


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedDocument],
    top_k: int = 10,
    k: int = 60,
) -> list[RetrievedDocument]:
    """
    Combine ranked retrieval lists using RRF.

    Formula:

        RRF(d) = sum(1 / (k + rank))

    Higher score means better combined ranking.
    """

    if top_k <= 0:
        return []

    if k <= 0:
        k = 60

    logger.info(
        "Running Reciprocal Rank Fusion | lists={} | top_k={} | k={}",
        len(ranked_lists),
        top_k,
        k,
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

            key = _document_key(document)

            if not key:
                continue

            scores[key] += (
                1.0 / (k + rank)
            )

            if key not in documents:
                documents[key] = document

    fused = sorted(
        documents.values(),
        key=lambda document: (
            scores[
                _document_key(document)
            ]
        ),
        reverse=True,
    )

    result = fused[:top_k]

    logger.info(
        "RRF completed | unique_documents={} | returned={}",
        len(fused),
        len(result),
    )

    return result