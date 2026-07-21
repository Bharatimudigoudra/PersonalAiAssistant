"""
Common API response models.
"""

from typing import Any

from pydantic import BaseModel


class APIResponse(BaseModel):
    """
    Standard API response.
    """

    success: bool
    message: str
    data: Any | None = None


class APIError(BaseModel):
    """
    Standard error response.
    """

    success: bool = False
    error: dict