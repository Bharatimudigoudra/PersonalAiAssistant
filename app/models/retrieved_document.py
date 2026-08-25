"""
Retrieved document model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    """
    Represents a document chunk returned by retrieval.
    """

    content: str

    # Chroma distance.
    # Lower distance means more similar.
    distance: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # CrossEncoder score.
    # Higher score means more relevant.
    rerank_score: float = 0.0