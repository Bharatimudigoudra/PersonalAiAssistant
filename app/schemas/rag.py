"""
RAG API schemas.
"""

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )


class RAGResponse(BaseModel):
    answer: str