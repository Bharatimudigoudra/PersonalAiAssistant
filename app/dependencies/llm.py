"""
LLM Dependency.

Provides a singleton LLMService instance
for FastAPI dependency injection.
"""

from functools import lru_cache

from app.llm.services import LLMService


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """
    Return a singleton LLMService instance.
    """

    return LLMService()