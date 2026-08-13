"""
Application-level LLM service.

Architecture:

InterviewAssistant
        ↓
LLMService
        ↓
LLMProviderFactory
        ↓
OllamaProvider
        ↓
Ollama

Responsibilities:
- Normal conversational generation
- RAG generation
- Conversation memory
- Streaming
- Provider health checks
- Backward-compatible prompt handling
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
    Single application-level service for all LLM operations.
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

    # ==============================================================
    # Prompt Construction
    # ==============================================================

    @staticmethod
    def _combine_prompts(
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """
        Build one prompt for the provider.

        The current BaseLLMProvider/OllamaProvider interface accepts
        one prompt plus optional conversation history.

        Therefore, system_prompt and user_prompt are combined here.
        """

        if user_prompt is not None:
            main_prompt = user_prompt.strip()
        elif prompt is not None:
            main_prompt = prompt.strip()
        else:
            main_prompt = ""

        if not main_prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if system_prompt:
            system = system_prompt.strip()

            if system:
                return (
                    "SYSTEM INSTRUCTIONS:\n"
                    "--------------------\n"
                    f"{system}\n\n"
                    "USER REQUEST:\n"
                    "-------------\n"
                    f"{main_prompt}"
                )

        return main_prompt

    # ==============================================================
    # Normal Generation
    # ==============================================================

    def generate(
        self,
        prompt: str | None = None,
        history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """
        Generate a normal conversational response.

        Supported usage:

            generate(
                prompt="Hello"
            )

        or:

            generate(
                system_prompt="You are an interviewer.",
                user_prompt="Ask me a question."
            )

        If history is omitted, application memory is used.
        """

        final_prompt = self._combine_prompts(
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        use_memory = history is None

        if use_memory:

            logger.info(
                "Saving user message to conversation memory."
            )

            self._memory.add_user_message(
                final_prompt
            )

            history = self._memory.history()

            logger.info(
                "Conversation contains {} messages.",
                len(history),
            )

        response = self._provider.generate(
            prompt=final_prompt,
            history=history,
        )

        response = (
            response.strip()
            if response
            else ""
        )

        if use_memory and response:

            logger.info(
                "Saving assistant response to conversation memory."
            )

            self._memory.add_assistant_message(
                response
            )

        return response

    # ==============================================================
    # RAG Generation
    # ==============================================================

    def generate_rag(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """
        Generate a RAG response.

        IMPORTANT:
        RAG generation intentionally bypasses conversation memory.

        The prompt already contains retrieved document context.
        """

        final_prompt = self._combine_prompts(
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.info(
            "Generating RAG response."
        )

        response = self._provider.generate(
            prompt=final_prompt,
            history=None,
        )

        response = (
            response.strip()
            if response
            else ""
        )

        return response

    # ==============================================================
    # Streaming
    # ==============================================================

    def stream(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> Iterator[str]:
        """
        Stream a response from the configured provider.

        Streaming intentionally does not use conversation memory.
        """

        final_prompt = self._combine_prompts(
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.info(
            "Starting LLM streaming."
        )

        yield from self._provider.stream(
            prompt=final_prompt,
            history=None,
        )

    # ==============================================================
    # Memory
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
        Clear conversation history.
        """

        logger.info(
            "Clearing conversation history."
        )

        self._memory.clear()

    # ==============================================================
    # Health Check
    # ==============================================================

    def health_check(
        self,
    ) -> bool:
        """
        Check whether the configured LLM provider is healthy.
        """

        return self._provider.health_check()


# ==============================================================
# Singleton
# ==============================================================

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