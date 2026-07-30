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
from app.memory.memory import ChatMessage
from app.memory.memory_service import (
    get_memory_service,
)


class LLMService:
    """
    Service layer for all LLM interactions.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
    ) -> None:

        self._provider = (
            provider
            or LLMProviderFactory.create()
        )

        self._memory = get_memory_service()

        logger.info(
            "LLMService initialized with {}",
            self._provider.__class__.__name__,
        )

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Generate a conversational response.

        If history is None,
        conversation memory is used.

        Otherwise the supplied history
        is used directly.
        """

        use_memory = history is None

        if use_memory:

            logger.info(
                "Saving user message."
            )

            self._memory.add_user_message(
                prompt,
            )

            history = self._memory.history()

            logger.info(
                "Conversation contains {} messages.",
                len(history),
            )

        else:

            logger.info(
                "Using supplied history ({} messages).",
                len(history),
            )

        response = self._provider.generate(
            prompt=prompt,
            history=history,
        )

        if use_memory:

            logger.info(
                "Saving assistant message."
            )

            self._memory.add_assistant_message(
                response,
            )

        return response

    def generate_rag(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response for RAG.

        Conversation memory is intentionally
        bypassed.
        """

        logger.info(
            "Generating RAG response."
        )

        return self._provider.generate(
            prompt=prompt,
            history=None,
        )

    def get_history(
        self,
    ) -> list[ChatMessage]:
        """
        Return conversation history.
        """

        return self._memory.history()

    def clear_history(
        self,
    ) -> None:
        """
        Clear conversation history.
        """

        logger.info(
            "Clearing conversation history."
        )

        self._memory.clear()

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        """
        Stream model output.
        """

        yield from self._provider.stream(
            prompt=prompt,
        )

    def health_check(
        self,
    ) -> bool:
        """
        Check provider health.
        """

        return self._provider.health_check()