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
            prompt: str | None = None,
            system_prompt: str | None = None,
            user_prompt: str | None = None,
        ) -> str:
            """
            Generate a response from the local LLM.

            Supports both styles:

                generate(prompt="...")

            and

                generate(
                    system_prompt="...",
                    user_prompt="...",
                )
            """

            # ---------------------------------------------
            # Backward compatibility
            # ---------------------------------------------
            if user_prompt is not None:
                prompt = user_prompt

            if prompt is None:
                raise ValueError(
                    "Either 'prompt' or 'user_prompt' must be provided."
                )

            logger.info("Generating response...")

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

                elapsed = time.perf_counter() - start

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
            prompt: str | None = None,
            system_prompt: str | None = None,
            user_prompt: str | None = None,
        ):
            """
            Stream tokens from Ollama.

            Supports both:

                stream(prompt="...")

            and

                stream(
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

            logger.info("Streaming response...")

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