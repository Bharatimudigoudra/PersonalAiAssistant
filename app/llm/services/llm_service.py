"""
Application-level LLM service.

Responsibilities:
- Normal conversational generation
- RAG generation
- Streaming
- Conversation memory
- Provider health checks

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
    Application-level service for LLM operations.

    This class does NOT construct interview prompts.
    PromptBuilder owns interview prompt construction.

    This class also does NOT clean model reasoning.
    OllamaProvider owns provider-specific response handling.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
    ) -> None:

        logger.info(
            "Creating LLM provider from factory."
        )

        self._provider = (
            provider
            if provider is not None
            else LLMProviderFactory.create()
        )

        self._memory = get_memory_service()

        logger.info(
            "LLMService initialized with provider={}",
            self._provider.__class__.__name__,
        )

    # ==============================================================
    # NORMAL GENERATION
    # ==============================================================

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Generate a normal conversational response.

        If history is not supplied, application memory is used.
        """

        if prompt is None:
            raise ValueError(
                "Prompt cannot be None."
            )

        prompt = str(prompt).strip()

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

        logger.info(
            "Generating normal response | prompt_chars={}",
            len(prompt),
        )

        response = self._provider.generate(
            prompt=prompt,
            history=history,
        )

        response = (
            str(response or "")
            .strip()
        )

        if use_memory and response:

            self._memory.add_assistant_message(
                response
            )

        logger.info(
            "Normal response generated | chars={}",
            len(response),
        )

        return response

    # ==============================================================
    # RAG GENERATION
    # ==============================================================

    def generate_rag(
        self,
        prompt: str,
    ) -> str:
        """
        Generate an interview answer from a fully constructed
        RAG prompt.

        IMPORTANT:
        RAG generation intentionally does not use conversation memory.

        The PromptBuilder has already constructed the complete prompt
        containing:
            - reference context
            - interview question
            - answer requirements
        """

        if prompt is None:
            raise ValueError(
                "RAG prompt cannot be None."
            )

        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError(
                "RAG prompt cannot be empty."
            )

        logger.info(
            "Generating RAG response | prompt_chars={}",
            len(prompt),
        )

        response = self._provider.generate(
            prompt=prompt,
            history=None,
        )

        response = (
            str(response or "")
            .strip()
        )

        logger.info(
            "RAG response generated | chars={}",
            len(response),
        )

        return response

    # ==============================================================
    # STREAMING
    # ==============================================================

    def stream(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[str]:
        """
        Stream a response from the configured provider.
        """

        if prompt is None:
            raise ValueError(
                "Prompt cannot be None."
            )

        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Starting LLM stream | prompt_chars={}",
            len(prompt),
        )

        yield from self._provider.stream(
            prompt=prompt,
            history=history,
        )

    # ==============================================================
    # MEMORY
    # ==============================================================

    def get_history(
        self,
    ) -> list[ChatMessage]:
        """
        Return current conversation history.
        """

        return self._memory.history()

    def clear_history(
        self,
    ) -> None:
        """
        Clear current conversation history.
        """

        logger.info(
            "Clearing conversation history."
        )

        self._memory.clear()

    # ==============================================================
    # HEALTH CHECK
    # ==============================================================

    def health_check(
        self,
    ) -> bool:
        """
        Check whether the configured provider is healthy.
        """

        try:

            healthy = (
                self._provider.health_check()
            )

            logger.info(
                "LLM provider health check | healthy={}",
                healthy,
            )

            return bool(healthy)

        except Exception:

            logger.exception(
                "LLM provider health check failed."
            )

            return False


# ==============================================================
# SINGLETON
# ==============================================================

_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """
    Return the application-level LLMService singleton.
    """

    global _service

    if _service is None:

        logger.info(
            "Creating application LLMService singleton."
        )

        _service = LLMService()

    return _service