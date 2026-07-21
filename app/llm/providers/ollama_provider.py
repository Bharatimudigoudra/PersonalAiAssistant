"""
Ollama provider implementation.

This provider communicates with the local Ollama server using
LangChain's ChatOllama integration.
"""

import time
from typing import Iterator

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.core.config import llm
from app.core.logging import logger
from app.llm.providers.base_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Concrete implementation of BaseLLMProvider using Ollama.
    """

    def __init__(self) -> None:
        logger.info("Initializing Ollama Provider...")

        self._llm = ChatOllama(
            model=llm.model_name,
            base_url=llm.base_url,
            temperature=llm.temperature,
            num_predict=1024,
            num_ctx=4096,
        )

        logger.info("Model Loaded: {}", llm.model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate a complete response.
        """

        logger.info("Generating response...")

        start = time.time()

        response = self._llm.invoke(
            [HumanMessage(content=prompt)]
        )

        elapsed = time.time() - start

        logger.info(
            "LLM completed in {:.2f} seconds.",
            elapsed,
        )

        # ==============================
        # RAW DEBUG
        # ==============================

        print("\n" + "=" * 80)
        print("RAW RESPONSE OBJECT")
        print("=" * 80)
        print(response)

        print("\n" + "=" * 80)
        print("RESPONSE TYPE")
        print("=" * 80)
        print(type(response))

        print("\n" + "=" * 80)
        print("__dict__")
        print("=" * 80)
        print(response.__dict__)

        print("\n" + "=" * 80)
        print("CONTENT")
        print("=" * 80)
        print(repr(response.content))

        if hasattr(response, "additional_kwargs"):
            print("\n" + "=" * 80)
            print("ADDITIONAL KWARGS")
            print("=" * 80)
            print(response.additional_kwargs)

        if hasattr(response, "response_metadata"):
            print("\n" + "=" * 80)
            print("RESPONSE METADATA")
            print("=" * 80)
            print(response.response_metadata)

        print("=" * 80 + "\n")

        # ==============================

        return response.content or ""

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the model response.
        """

        logger.info("Streaming response...")

        start = time.time()

        for chunk in self._llm.stream(
            [HumanMessage(content=prompt)]
        ):
            if chunk.content:
                yield chunk.content

        elapsed = time.time() - start

        logger.info(
            "Streaming completed in {} seconds.",
            round(elapsed, 2),
        )

    def health_check(self) -> bool:
        """
        Check whether Ollama is available.
        """

        try:
            self.generate("Hello")

            logger.info("Ollama health check passed.")

            return True

        except Exception as exc:
            logger.exception(
                "Ollama health check failed: {}",
                exc,
            )
            return False