"""
Abstract base class for all LLM providers.

Every provider (Ollama, llama.cpp, etc.) must implement
this interface so the rest of the application can interact
with any LLM in a consistent way.
"""

from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLMProvider(ABC):
    """
    Defines the contract for all LLM providers.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a complete response.

        Args:
            prompt: User input prompt.

        Returns:
            Model response as a string.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the response token-by-token.

        Args:
            prompt: User input prompt.

        Yields:
            Individual text chunks from the model.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check whether the LLM service is available.

        Returns:
            True if the provider is healthy.
        """
        raise NotImplementedError