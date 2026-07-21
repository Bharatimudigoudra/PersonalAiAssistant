"""
Ollama provider implementation.

This provider communicates with the local Ollama server using
LangChain's ChatOllama integration.
"""

import time
from typing import Iterator

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_ollama import ChatOllama

from app.core.config import llm
from app.core.logging import logger
from app.llm.providers.base_provider import BaseLLMProvider
from app.memory.models import ChatMessage


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
            num_predict=512,
            num_ctx=2048,
        )

        logger.info(
            "Model Loaded: {}",
            llm.model_name,
        )

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Generate a complete response.
        """

        logger.info("Generating response...")

        start = time.time()

        messages = []

        if history:

            logger.info(
                "Loading {} history messages.",
                len(history),
            )

            for message in history:

                if message.role == "user":

                    messages.append(
                        HumanMessage(
                            content=message.content,
                        )
                    )

                elif message.role == "assistant":

                    messages.append(
                        AIMessage(
                            content=message.content,
                        )
                    )

        else:

            messages.append(
                HumanMessage(
                    content=prompt,
                )
            )

        response = self._llm.invoke(messages)

        elapsed = time.time() - start

        logger.info(
            "LLM completed in {:.2f} seconds.",
            elapsed,
        )

        return response.content or ""

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        """
        Stream the model response.
        """

        logger.info("Streaming response...")

        start = time.time()

        for chunk in self._llm.stream(
            [
                HumanMessage(
                    content=prompt,
                )
            ]
        ):
            if chunk.content:
                yield chunk.content

        elapsed = time.time() - start

        logger.info(
            "Streaming completed in {:.2f} seconds.",
            elapsed,
        )

    def health_check(
        self,
    ) -> bool:
        """
        Check whether Ollama is available.
        """

        try:

            self.generate("Hello")

            logger.info(
                "Ollama health check passed."
            )

            return True

        except Exception as exc:

            logger.exception(
                "Ollama health check failed: {}",
                exc,
            )

            return False