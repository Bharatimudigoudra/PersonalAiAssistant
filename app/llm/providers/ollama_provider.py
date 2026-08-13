"""
Ollama LLM provider.

Local Ollama provider used by PersonalAiAssistant.

Responsibilities:
- Connect to Ollama
- Generate non-streaming responses
- Stream responses
- Disable Qwen3 thinking when supported
- Keep interview responses concise
- Safely extract response content
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
    """LLM provider backed by a local Ollama server."""

    def __init__(self) -> None:
        logger.info("Initializing Ollama Provider...")

        self._client = ollama.Client(
            host=llm.base_url,
        )

        self.model = llm.model_name

        logger.info(
            "Ollama Provider initialized | model={}",
            self.model,
        )

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build Ollama chat messages.

        History is intentionally included only when supplied by the
        service layer.
        """

        messages: list[dict[str, str]] = []

        if history:
            for message in history:
                role = message.role

                # Ollama expects standard roles.
                if role not in {
                    "system",
                    "user",
                    "assistant",
                }:
                    role = "user"

                messages.append(
                    {
                        "role": role,
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

    # ------------------------------------------------------------------
    # Ollama options
    # ------------------------------------------------------------------

    @staticmethod
    def _options() -> dict:
        """
        Runtime options optimized for a local interview assistant.
        """

        return {
            "temperature": float(llm.temperature),

            # Keep prompt/context manageable on local hardware.
            "num_ctx": 4096,

            # Interview answers should be short.
            "num_predict": min(
                int(llm.max_tokens),
                256,
            ),

            # Helps reduce unnecessary repetition.
            "repeat_penalty": 1.1,
        }

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(response) -> str:
        """Safely extract text from an Ollama response."""

        if response is None:
            return ""

        try:
            message = response.get("message", {})

            if not isinstance(message, dict):
                return ""

            content = message.get("content", "")

            if content is None:
                return ""

            return str(content).strip()

        except Exception:
            logger.exception(
                "Failed to extract Ollama response."
            )
            return ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response(text: str) -> str:
        """
        Remove accidental reasoning wrappers.

        Qwen3 should run with think=False, but this defensive cleanup
        prevents internal-looking sections from reaching the interviewer.
        """

        if not text:
            return ""

        text = text.strip()

        # Remove common explicit thinking blocks if returned.
        markers = [
            "</think>",
            "<|endofthink|>",
        ]

        for marker in markers:
            if marker in text:
                text = text.split(marker)[-1].strip()

        # If an opening thinking marker exists but no closing marker,
        # remove everything before the final likely answer.
        if "<think>" in text:
            parts = text.split("</think>", 1)

            if len(parts) == 2:
                text = parts[1].strip()

        return text.strip()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Generate a complete response."""

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Generating response with model={}",
            self.model,
        )

        logger.info(
            "Prompt length={} characters",
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

                # Qwen3: disable reasoning for interview responses.
                think=False,

                stream=False,

                options=self._options(),
            )

        except Exception:
            elapsed = time.perf_counter() - start

            logger.exception(
                "Ollama generation failed after {:.2f}s",
                elapsed,
            )

            return ""

        elapsed = time.perf_counter() - start

        content = self._extract_content(response)

        content = self._clean_response(content)

        if not content:
            logger.warning(
                "Ollama returned an empty response."
            )

            return ""

        logger.info(
            "Generation completed in {:.2f}s",
            elapsed,
        )

        logger.info(
            "Generated {} characters",
            len(content),
        )

        return content

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[str]:
        """Stream Ollama output."""

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Streaming response with model={}",
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
                options=self._options(),
            )

            for chunk in response_stream:
                token = ""

                try:
                    message = chunk.get(
                        "message",
                        {},
                    )

                    if isinstance(message, dict):
                        token = message.get(
                            "content",
                            "",
                        )

                except Exception:
                    logger.exception(
                        "Failed to read Ollama stream chunk."
                    )

                if token:
                    total_chars += len(token)
                    yield token

            elapsed = time.perf_counter() - start

            logger.info(
                "Streaming completed in {:.2f}s",
                elapsed,
            )

            logger.info(
                "Streamed {} characters",
                total_chars,
            )

        except Exception:
            elapsed = time.perf_counter() - start

            logger.exception(
                "Ollama streaming failed after {:.2f}s",
                elapsed,
            )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check whether Ollama can generate a response."""

        try:
            response = self.generate(
                prompt=(
                    "Reply with exactly one word: OK"
                )
            )

            healthy = response.strip().upper() == "OK"

            if healthy:
                logger.info(
                    "Ollama health check passed."
                )
            else:
                logger.warning(
                    "Ollama health check failed. Response={!r}",
                    response,
                )

            return healthy

        except Exception:
            logger.exception(
                "Ollama health check failed."
            )

            return False