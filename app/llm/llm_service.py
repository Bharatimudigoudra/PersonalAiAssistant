"""
LLM Service.

Provides a singleton interface for the local Ollama LLM.

Designed for:
- Ollama local inference
- Qwen3 models
- Non-thinking interview responses
- Streaming and non-streaming generation
- Robust response extraction
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import ollama

from app.core.config import llm
from app.core.logging import logger


# ---------------------------------------------------------------------
# Singleton Ollama Client
# ---------------------------------------------------------------------

_client: ollama.Client | None = None


class LLMService:
    """
    High-level wrapper around a local Ollama model.
    """

    def __init__(self) -> None:
        global _client

        if _client is None:
            logger.info(
                "Connecting to Ollama at {}",
                llm.base_url,
            )

            _client = ollama.Client(
                host=llm.base_url,
            )

        self.client = _client
        self.model = llm.model_name

        logger.info(
            "LLMService initialized (model={}).",
            self.model,
        )

    # -----------------------------------------------------------------
    # Message Builder
    # -----------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build Ollama chat messages.
        """

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
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
    # Response Extraction
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_content(response) -> str:
        """
        Safely extract generated text from an Ollama response.
        """

        try:
            message = response.get("message", {})

            content = message.get("content", "")

            if content is None:
                return ""

            return str(content).strip()

        except Exception:
            logger.exception(
                "Failed to extract Ollama response content."
            )

            return ""

    # -----------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------

    def generate(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """
        Generate a complete response from Ollama.

        Supports:

            generate(prompt="...")

        or:

            generate(
                system_prompt="...",
                user_prompt="...",
            )
        """

        if user_prompt is not None:
            prompt = user_prompt

        if prompt is None:
            raise ValueError(
                "Either 'prompt' or 'user_prompt' must be provided."
            )

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
            "Prompt length: {} characters.",
            len(prompt),
        )

        start = time.perf_counter()

        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,

                # IMPORTANT FOR QWEN3:
                # Disable internal thinking for interview answers.
                think=False,

                # Non-streaming generation.
                stream=False,

                options={
                    "temperature": llm.temperature,

                    # Keep the context reasonable for local CPU inference.
                    "num_ctx": 4096,

                    # Do NOT generate thousands of tokens for an
                    # interview answer.
                    "num_predict": min(
                        int(llm.max_tokens),
                        512,
                    ),
                },
            )

            answer = self._extract_content(response)

            elapsed = time.perf_counter() - start

            if not answer:
                logger.warning(
                    "Ollama returned an empty response."
                )

                logger.warning(
                    "Raw Ollama response type: {}",
                    type(response).__name__,
                )

                return ""

            logger.info(
                "Generation completed in {:.2f} sec.",
                elapsed,
            )

            logger.info(
                "Generated {} characters.",
                len(answer),
            )

            return answer

        except Exception:
            elapsed = time.perf_counter() - start

            logger.exception(
                "LLM generation failed after {:.2f} sec.",
                elapsed,
            )

            return ""

    # -----------------------------------------------------------------
    # Stream
    # -----------------------------------------------------------------

    def stream(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens from Ollama.
        """

        if user_prompt is not None:
            prompt = user_prompt

        if prompt is None:
            raise ValueError(
                "Either 'prompt' or 'user_prompt' must be provided."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Streaming response with model={}",
            self.model,
        )

        logger.info(
            "Prompt length: {} characters.",
            len(prompt),
        )

        start = time.perf_counter()
        total_chars = 0

        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        try:
            response_stream = self.client.chat(
                model=self.model,
                messages=messages,

                # IMPORTANT FOR QWEN3.
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

                try:
                    message = chunk.get(
                        "message",
                        {},
                    )

                    token = message.get(
                        "content",
                        "",
                    )

                except Exception:
                    logger.exception(
                        "Failed to read Ollama stream chunk."
                    )
                    continue

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

            if total_chars == 0:
                logger.warning(
                    "Ollama streaming completed with zero characters."
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
        """
        Verify that Ollama is available and can generate text.
        """

        try:

            response = self.generate(
                prompt="Reply with exactly: OK"
            )

            healthy = response.strip() == "OK"

            if healthy:
                logger.info(
                    "Ollama health check passed."
                )
            else:
                logger.warning(
                    "Ollama health check returned: {!r}",
                    response,
                )

            return healthy

        except Exception:

            logger.exception(
                "Ollama health check failed."
            )

            return False


# ---------------------------------------------------------------------
# Singleton Service
# ---------------------------------------------------------------------

_llm_service = LLMService()


def get_llm_service() -> LLMService:
    """
    Return the singleton LLM service.
    """

    return _llm_service


# ---------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    service = get_llm_service()

    print()
    print("=" * 70)
    print("Testing Ollama LLM")
    print("=" * 70)

    reply = service.generate(
        prompt=(
            "Tell me about yourself in 3 short sentences."
        )
    )

    print()
    print(reply)
    print()
    print("=" * 70)