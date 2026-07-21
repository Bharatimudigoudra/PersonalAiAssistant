"""
LLM Service.

Provides a unified interface for interacting with the configured
language model provider.
"""

from typing import Iterator

from app.core.logging import logger
from app.llm.providers import BaseLLMProvider, LLMProviderFactory


class LLMService:
    """
    Service layer for LLM interactions.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        """
        Initialize the service.

        Args:
            provider:
                Optional provider for dependency injection.
                If omitted, the configured provider is created
                using the factory.
        """

        self._provider = provider or LLMProviderFactory.create()

        logger.info(
            "LLMService initialized with {}",
            self._provider.__class__.__name__,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a complete response.
        """

        return self._provider.generate(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the response.
        """

        yield from self._provider.stream(prompt)

    def health_check(self) -> bool:
        """
        Check provider health.
        """

        return self._provider.health_check()