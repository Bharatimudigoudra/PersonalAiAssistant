"""
LLM Provider Factory.

Creates and returns the configured LLM provider.
"""

from app.core.config import llm
from app.core.logging import logger
from app.llm.providers.base_provider import BaseLLMProvider
from app.llm.providers.ollama_provider import OllamaProvider


class LLMProviderFactory:
    """
    Factory responsible for creating LLM providers.
    """

    @staticmethod
    def create() -> BaseLLMProvider:
        """
        Create the configured LLM provider.

        Returns:
            BaseLLMProvider
        """

        provider = llm.provider.lower()

        logger.info("Selected LLM Provider: {}", provider)

        if provider == "ollama":
            return OllamaProvider()

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )