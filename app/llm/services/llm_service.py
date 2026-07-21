"""
LLM Service.

Provides a unified interface for interacting with the configured
language model provider.
"""

from typing import Iterator

from app.core.logging import logger
from app.llm.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
)
from app.memory.service import MemoryService


class LLMService:
    """
    Service layer for LLM interactions.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        """
        Initialize the LLM service.
        """

        self._provider = provider or LLMProviderFactory.create()

        self.memory = MemoryService()

        logger.info(
            "LLMService initialized with {}",
            self._provider.__class__.__name__,
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response while maintaining conversation memory.
        """

        logger.info("Saving user message...")

        self.memory.add_user_message(prompt)

        history = self.memory.get_history()

        logger.info(
            "Conversation contains {} messages.",
            len(history),
        )

        response = self._provider.generate(
            prompt,
            history,
        )

        logger.info("Saving assistant message...")

        self.memory.add_assistant_message(
            response,
        )

        return response

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        """
        Stream the response.
        """

        yield from self._provider.stream(prompt)

    def health_check(
        self,
    ) -> bool:
        """
        Check provider health.
        """

        return self._provider.health_check()