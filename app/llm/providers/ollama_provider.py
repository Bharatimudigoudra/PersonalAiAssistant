"""
Ollama LLM provider.

Local inference provider for Ollama.

Designed for:
- Qwen3
- interview answers
- RAG generation
- non-thinking responses
- streaming
- defensive removal of leaked reasoning
"""

from __future__ import annotations

import re
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

        A strict system instruction is added so the model behaves
        like an interview-answer generator instead of explaining
        how it generated the answer.
        """

        messages: list[dict[str, str]] = []

        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a professional interview answer assistant. "
                    "Answer the interviewer's question directly. "
                    "Speak in first person when answering questions about "
                    "the candidate. "
                    "Use only the information provided in the user prompt. "
                    "Do not explain your reasoning. "
                    "Do not describe the context. "
                    "Do not say 'we are given the context'. "
                    "Do not mention documents, retrieved context, prompts, "
                    "instructions, or RAG. "
                    "Do not produce analysis or thinking. "
                    "Return only the final interview answer."
                ),
            }
        )

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

    # ------------------------------------------------------------------
    # Response cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response(
        content: str | None,
    ) -> str:
        """
        Remove leaked reasoning and internal meta text.

        Handles both:

        1. Ollama's separate thinking field.
        2. Qwen-style <think>...</think> content leakage.
        """

        if not content:
            return ""

        text = str(content).strip()

        # --------------------------------------------------------------
        # Remove complete <think>...</think> blocks
        # --------------------------------------------------------------

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # --------------------------------------------------------------
        # If only the closing tag is present, keep everything after it.
        # Example:
        #
        # reasoning...
        # </think>
        #
        # Final answer
        # --------------------------------------------------------------

        if "</think>" in text.lower():

            parts = re.split(
                r"</think>",
                text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )

            if len(parts) == 2:

                text = parts[1]

        # --------------------------------------------------------------
        # Remove orphan thinking tags
        # --------------------------------------------------------------

        text = re.sub(
            r"</?think>",
            "",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

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
            "Generating response with model={}",
            self.model,
        )

        logger.info(
            "Prompt length={} characters",
            len(prompt),
        )

        messages = self._build_messages(
            prompt=prompt,
            history=history,
        )

        logger.info(
            "Sending {} messages to Ollama.",
            len(messages),
        )

        start = time.perf_counter()

        try:

            response = self._client.chat(
                model=self.model,
                messages=messages,

                # IMPORTANT:
                # think must be passed as a top-level parameter.
                think=False,

                stream=False,

                options={
                    "temperature": float(
                        llm.temperature
                    ),

                    "num_ctx": 4096,

                    # Keep interview answers short.
                    "num_predict": min(
                        int(llm.max_tokens),
                        256,
                    ),
                },
            )

        except Exception:

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.exception(
                "Ollama request failed after {:.2f} sec.",
                elapsed,
            )

            return ""

        elapsed = (
            time.perf_counter()
            - start
        )

        # --------------------------------------------------------------
        # Extract message
        # --------------------------------------------------------------

        try:

            message = response.get(
                "message",
                {},
            )

        except Exception:

            logger.exception(
                "Unable to read Ollama response."
            )

            return ""

        # --------------------------------------------------------------
        # Thinking is intentionally ignored.
        #
        # Ollama's current API exposes reasoning separately as
        # message.thinking.
        # --------------------------------------------------------------

        thinking = ""

        try:

            thinking = (
                message.get(
                    "thinking",
                    "",
                )
                or ""
            )

        except Exception:

            thinking = ""

        if thinking:

            logger.debug(
                "Ollama returned {} thinking characters; "
                "thinking will not be exposed.",
                len(str(thinking)),
            )

        # --------------------------------------------------------------
        # Final answer
        # --------------------------------------------------------------

        content = message.get(
            "content",
            "",
        )

        content = self._clean_response(
            content
        )

        logger.info(
            "Ollama request completed in {:.2f} sec.",
            elapsed,
        )

        logger.info(
            "Generated {} characters.",
            len(content),
        )

        if not content:

            logger.warning(
                "Ollama returned an empty final response."
            )

            return ""

        return content

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

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
            "Streaming response with model={}",
            self.model,
        )

        messages = self._build_messages(
            prompt=prompt,
            history=history,
        )

        start = time.perf_counter()
        total_chars = 0

        try:

            response_stream = self._client.chat(
                model=self.model,
                messages=messages,

                think=False,

                stream=True,

                options={
                    "temperature": float(
                        llm.temperature
                    ),
                    "num_ctx": 4096,
                    "num_predict": min(
                        int(llm.max_tokens),
                        256,
                    ),
                },
            )

            for chunk in response_stream:

                try:

                    message = chunk.get(
                        "message",
                        {},
                    )

                    # Never expose thinking.
                    thinking = message.get(
                        "thinking",
                        "",
                    )

                    if thinking:
                        continue

                    token = message.get(
                        "content",
                        "",
                    )

                except Exception:

                    logger.exception(
                        "Failed to read Ollama stream chunk."
                    )

                    continue

                if not token:
                    continue

                # Defensive handling if the model still
                # sends <think> tags in the stream.
                token = re.sub(
                    r"</?think>",
                    "",
                    str(token),
                    flags=re.IGNORECASE,
                )

                if token:

                    total_chars += len(token)

                    yield token

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Streaming completed in {:.2f} sec.",
                elapsed,
            )

            logger.info(
                "Generated {} characters.",
                total_chars,
            )

        except Exception:

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.exception(
                "Streaming failed after {:.2f} sec.",
                elapsed,
            )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:

        try:

            response = self.generate(
                prompt=(
                    "Reply with exactly one word: OK"
                )
            )

            healthy = (
                response.strip().upper()
                == "OK"
            )

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