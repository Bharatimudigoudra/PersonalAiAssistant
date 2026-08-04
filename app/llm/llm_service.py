"""
LLM Service.

Provides a singleton interface for the local Ollama LLM.
"""

from __future__ import annotations

import time

import ollama

from app.core.config import llm
from app.core.logging import logger


# ---------------------------------------------------------------------
# Singleton Ollama Client
# ---------------------------------------------------------------------

_client: ollama.Client | None = None


class LLMService:
    """
    Wrapper around a local Ollama model.
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

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a complete response.
        """

        logger.info(
            "Generating response..."
        )

        start = time.perf_counter()

        messages = []

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

        try:

            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={
                    "temperature": llm.temperature,
                    "num_predict": llm.max_tokens,
                },
            )

            answer = response["message"]["content"].strip()

            elapsed = (
                time.perf_counter()
                - start
            )

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

            logger.exception(
                "LLM generation failed."
            )

            return ""

    # -----------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ):
        """
        Stream tokens from Ollama.
        """

        logger.info(
            "Streaming response..."
        )

        messages = []

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

        try:

            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": llm.temperature,
                    "num_predict": llm.max_tokens,
                },
            )

            for chunk in stream:

                yield chunk["message"]["content"]

        except Exception:

            logger.exception(
                "Streaming failed."
            )

            yield ""


# ---------------------------------------------------------------------
# Singleton Service
# ---------------------------------------------------------------------

_llm_service = LLMService()


def get_llm_service() -> LLMService:
    """
    Return singleton LLM service.
    """

    return _llm_service


# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    service = get_llm_service()

    reply = service.generate(
        prompt="Introduce yourself in one sentence."
    )

    print()
    print("=" * 60)
    print(reply)
    print("=" * 60)