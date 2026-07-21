"""
Embedding schemas.
"""

from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """
    Embedding response.
    """

    embedding: list[float]


class BatchEmbeddingResult(BaseModel):
    """
    Batch embedding response.
    """

    embeddings: list[list[float]]