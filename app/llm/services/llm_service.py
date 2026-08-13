"""
Application-level LLM service.

Single service responsible for:

- normal conversational generation
- RAG generation
- streaming
- conversation memory
- provider health checks

Architecture:

InterviewAssistant
        |
        v
LLMService
        |
        v
LLMProviderFactory
        |
        v
OllamaProvider
        |
        v
Ollama
"""

from __future__ import annotations

from collections.abc import Iterator

from app.core.logging import logger
from app.llm.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
)
from app.memory.memory import ChatMessage
from app.memory.memory_service import get_memory_service


class LLMService:
    """
    Single application-level LLM service.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
    ) -> None:

        self._provider = (
            provider
            if provider is not None
            else LLMProviderFactory.create()
        )

        self._memory = get_memory_service()

        logger.info(
            "LLMService initialized with {}",
            self._provider.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Normal generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Generate a normal conversational response.

        If history is None, application memory is used.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        use_memory = history is None

        if use_memory:

            self._memory.add_user_message(
                prompt
            )

            history = self._memory.history()

        response = self._provider.generate(
            prompt=prompt,
            history=history,
        )

        response = response.strip()

        if use_memory and response:

            self._memory.add_assistant_message(
                response
            )

        return response

    # ------------------------------------------------------------------
    # RAG generation
    # ------------------------------------------------------------------

    def generate_rag(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a RAG response.

        RAG generation intentionally bypasses
        conversation memory because the prompt
        already contains the retrieved context.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "RAG prompt cannot be empty."
            )

        logger.info(
            "Generating RAG response."
        )

        response = self._provider.generate(
            prompt=prompt,
            history=None,
        )

        return response.strip()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        """
        Stream a response from the configured provider.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        yield from self._provider.stream(
            prompt=prompt,
            history=None,
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(
        self,
    ) -> bool:
        """
        Check configured LLM provider.
        """

        return self._provider.health_check()


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """
    Return the application-level LLM service singleton.
    """

    global _service

    if _service is None:

        logger.info(
            "Creating application LLMService singleton."
        )

        _service = LLMService()

    return _service