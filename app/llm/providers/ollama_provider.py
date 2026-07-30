"""
Ollama provider implementation using the official Ollama Python client.
"""

import time
from typing import Iterator

import ollama

from app.core.config import llm
from app.core.logging import logger
from app.llm.providers.base_provider import BaseLLMProvider
from app.memory.memory import ChatMessage


class OllamaProvider(BaseLLMProvider):
    """
    Concrete implementation of BaseLLMProvider using Ollama.
    """

    def __init__(self) -> None:

        logger.info("Initializing Ollama Provider...")

        self._client = ollama.Client(
            host=llm.base_url,
        )

        logger.info(
            "Loaded model: {}",
            llm.model_name,
        )

    def _build_messages(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> list[dict]:
        """
        Build Ollama chat messages.
        """

        messages: list[dict] = []

        if history:

            for message in history:

                messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Generate a complete response.
        """

        logger.info("Generating response...")

        start = time.perf_counter()

        try:

            response = self._client.chat(
                model=llm.model_name,
                messages=self._build_messages(
                    prompt=prompt,
                    history=history,
                ),
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": 2048,
                    "num_predict": 150,
                },
            )

        except Exception as exc:

            logger.exception(
                "Ollama request failed: {}",
                exc,
            )

            return ""

        elapsed = time.perf_counter() - start

        logger.info(
            "LLM completed in {:.2f} seconds.",
            elapsed,
        )

        content = (
            response.get("message", {})
            .get("content", "")
            .strip()
        )

        logger.info(
            "Response length: {} characters.",
            len(content),
        )

        if not content:

            logger.warning(
                "LLM returned an empty response."
            )

            return ""

        return content

    def stream(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[str]:
        """
        Stream tokens from Ollama.
        """

        logger.info("Streaming response...")

        start = time.perf_counter()

        try:

            stream = self._client.chat(
                model=llm.model_name,
                messages=self._build_messages(
                    prompt=prompt,
                    history=history,
                ),
                think=False,
                stream=True,
                options={
                    "temperature": 0,
                    "num_ctx": 2048,
                    "num_predict": 150,
                },
            )

            total_chars = 0

            for chunk in stream:

                token = (
                    chunk.get("message", {})
                    .get("content", "")
                )

                if token:

                    total_chars += len(token)

                    yield token

            logger.info(
                "Streaming completed ({} chars, {:.2f} sec).",
                total_chars,
                time.perf_counter() - start,
            )

        except Exception as exc:

            logger.exception(
                "Streaming failed: {}",
                exc,
            )

            yield ""

    def health_check(
        self,
    ) -> bool:
        """
        Verify Ollama is available.
        """

        try:

            response = self.generate(
                prompt="Hello",
            )

            healthy = bool(response)

            if healthy:

                logger.info(
                    "Ollama health check passed."
                )

            else:

                logger.warning(
                    "Health check returned an empty response."
                )

            return healthy

        except Exception as exc:

            logger.exception(
                "Ollama health check failed: {}",
                exc,
            )

            return False