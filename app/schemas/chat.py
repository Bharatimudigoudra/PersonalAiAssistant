"""
Chat API schemas.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Chat request model.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User prompt",
    )


class ChatResponse(BaseModel):
    """
    Chat response model.
    """

    response: str