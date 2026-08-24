"""
Retrieved document model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    """
    Represents one document chunk returned by retrieval.
    """

    content: str

    # Chroma cosine distance.
    #
    # Lower is better:
    #   0.0  -> identical
    #   1.0  -> weak relationship
    #   2.0  -> opposite direction
    distance: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # CrossEncoder score.
    # Higher is generally better for the reranker.
    rerank_score: float | None = None

    def __post_init__(self) -> None:

        self.content = str(
            self.content or ""
        ).strip()

        self.distance = float(
            self.distance
        )

        if self.rerank_score is not None:
            self.rerank_score = float(
                self.rerank_score
            )

        if self.metadata is None:
            self.metadata = {}