"""
Ollama LLM provider.

Uses the official Ollama Python client for local inference.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import ollama

from app.core.config import llm
from app.core.logging import logger
from app.llm.providers.base_provider import BaseLLMProvider
from app.memory.memory import ChatMessage


class OllamaProvider(BaseLLMProvider):
    """
    Concrete LLM provider using local Ollama inference.
    """

    def __init__(self) -> None:

        logger.info(
            "Initializing Ollama Provider..."
        )

        self._client = ollama.Client(
            host=llm.base_url,
        )

        self.model = llm.model_name

        logger.info(
            "Loaded model: {}",
            self.model,
        )

    # -----------------------------------------------------------------
    # Message Builder
    # -----------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> list[dict[str, str]]:

        messages: list[dict[str, str]] = []

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

    # -----------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Generating response with {}...",
            self.model,
        )

        logger.info(
            "Prompt length: {} characters.",
            len(prompt),
        )

        start = time.perf_counter()

        try:

            response = self._client.chat(
                model=self.model,
                messages=self._build_messages(
                    prompt=prompt,
                    history=history,
                ),

                # Disable Qwen3 reasoning for fast interview answers.
                think=False,

                stream=False,

                options={
                    "temperature": llm.temperature,
                    "num_ctx": 4096,
                    "num_predict": min(
                        int(llm.max_tokens),
                        512,
                    ),
                },
            )

        except Exception:

            elapsed = time.perf_counter() - start

            logger.exception(
                "Ollama request failed after {:.2f} sec.",
                elapsed,
            )

            return ""

        elapsed = time.perf_counter() - start

        content = (
            response
            .get("message", {})
            .get("content", "")
        )

        if content is None:
            content = ""

        content = str(content).strip()

        logger.info(
            "LLM completed in {:.2f} sec.",
            elapsed,
        )

        logger.info(
            "Response length: {} characters.",
            len(content),
        )

        if not content:

            logger.warning(
                "Ollama returned an empty response."
            )

            return ""

        return content

    # -----------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[str]:

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Streaming response from {}...",
            self.model,
        )

        start = time.perf_counter()
        total_chars = 0

        try:

            response_stream = self._client.chat(
                model=self.model,
                messages=self._build_messages(
                    prompt=prompt,
                    history=history,
                ),

                think=False,

                stream=True,

                options={
                    "temperature": llm.temperature,
                    "num_ctx": 4096,
                    "num_predict": min(
                        int(llm.max_tokens),
                        512,
                    ),
                },
            )

            for chunk in response_stream:

                token = (
                    chunk
                    .get("message", {})
                    .get("content", "")
                )

                if token:

                    total_chars += len(token)

                    yield token

            elapsed = time.perf_counter() - start

            logger.info(
                "Streaming completed in {:.2f} sec.",
                elapsed,
            )

            logger.info(
                "Generated {} characters.",
                total_chars,
            )

        except Exception:

            elapsed = time.perf_counter() - start

            logger.exception(
                "Streaming failed after {:.2f} sec.",
                elapsed,
            )

    # -----------------------------------------------------------------
    # Health Check
    # -----------------------------------------------------------------

    def health_check(self) -> bool:

        try:

            response = self.generate(
                prompt="Reply with exactly: OK"
            )

            if response == "OK":

                logger.info(
                    "Ollama health check passed."
                )

                return True

            logger.warning(
                "Ollama health check returned: {!r}",
                response,
            )

            return False

        except Exception:

            logger.exception(
                "Ollama health check failed."
            )

            return False