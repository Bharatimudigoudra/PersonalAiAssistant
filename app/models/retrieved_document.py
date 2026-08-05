from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    """
    Represents one retrieved document chunk.
    """

    content: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float = 0.0