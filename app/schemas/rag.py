"""
RAG API schemas.
"""

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """
    Request model for Retrieval-Augmented Generation.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question to answer using retrieved documents.",
        examples=[
            "What projects has Bharati worked on?"
        ],
    )


class RAGResponse(BaseModel):
    """
    """

    answer: str = Field(
        ...,
        description="Generated answer.",
    )