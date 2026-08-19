"""
Ollama LLM Provider.

Local Ollama inference provider for PersonalAiAssistant.

Responsibilities:
- Connect to local Ollama
- Generate normal LLM responses
- Generate RAG interview answers
- Support conversation history
- Support streaming
- Disable Qwen thinking when supported
- Never expose Ollama thinking/reasoning
- Remove leaked <think> blocks defensively
- Provide Ollama health checks
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from typing import Any

import ollama

from app.core.config import llm
from app.core.logging import logger
from app.llm.providers.base_provider import BaseLLMProvider
from app.memory.memory import ChatMessage


class OllamaProvider(BaseLLMProvider):
    """
    Ollama implementation of BaseLLMProvider.
    """

    def __init__(self) -> None:
        logger.info("Initializing Ollama Provider...")

        self._client = ollama.Client(
            host=llm.base_url,
        )

        self.model = str(
            llm.model_name
        ).strip()

        if not self.model:
            raise ValueError(
                "LLM model name cannot be empty."
            )

        logger.info(
            "Ollama Provider initialized | model={}",
            self.model,
        )

    # ==================================================================
    # SYSTEM INSTRUCTION
    # ==================================================================

    @staticmethod
    def _system_instruction() -> str:
        """
        Stable system instruction for the interview assistant.

        Important:
        The retrieved RAG context remains inside the user prompt.
        """

        return (
            "You are Bharati's personal AI interview assistant.\n\n"

            "Your job is to write the answer that Bharati should "
            "say directly to an interviewer.\n\n"

            "STRICT RULES:\n"
            "1. Return ONLY the final answer.\n"
            "2. Never show your reasoning or analysis.\n"
            "3. Never mention prompts, documents, RAG, retrieval, "
            "context, references, or system instructions.\n"
            "4. Never say 'we are given the context'.\n"
            "5. Never say 'according to the documents'.\n"
            "6. Never describe how you selected the answer.\n"
            "7. Answer in first person as Bharati when the question "
            "is about Bharati.\n"
            "8. Use only facts supplied in the user message.\n"
            "9. Never invent experience, education, skills, projects, "
            "companies, dates, achievements, or technologies.\n"
            "10. Keep the answer natural and suitable for speaking "
            "during an interview.\n"
            "11. Prefer 2 to 6 sentences unless the question requires "
            "more detail.\n"
            "12. Do not use <think> tags.\n"
            "13. Do not output analysis before the answer.\n"
            "14. Do not output an 'Answer:' or 'Final Answer:' heading.\n\n"

            "If the supplied information is insufficient, respond only:\n"
            "\"I don't have enough information to answer that accurately.\""
        )

    # ==================================================================
    # MESSAGE BUILDING
    # ==================================================================

    @classmethod
    def _build_messages(
        cls,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the messages sent to Ollama.
        """

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": cls._system_instruction(),
            }
        ]

        if history:
            for item in history:

                role = str(
                    getattr(
                        item,
                        "role",
                        "user",
                    )
                ).strip().lower()

                if role not in {
                    "system",
                    "user",
                    "assistant",
                }:
                    role = "user"

                content = str(
                    getattr(
                        item,
                        "content",
                        "",
                    )
                ).strip()

                if not content:
                    continue

                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    # ==================================================================
    # RESPONSE EXTRACTION
    # ==================================================================

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:
        """
        Extract ONLY message.content.

        Ollama/Qwen may also return a 'thinking' field.
        That field is deliberately ignored.
        """

        if response is None:
            return ""

        try:
            # ----------------------------------------------------------
            # Ollama response object
            # ----------------------------------------------------------

            message = getattr(
                response,
                "message",
                None,
            )

            if message is not None:

                content = getattr(
                    message,
                    "content",
                    None,
                )

                if content is not None:
                    return str(
                        content
                    ).strip()

            # ----------------------------------------------------------
            # Dictionary response
            # ----------------------------------------------------------

            if isinstance(
                response,
                dict,
            ):

                message = response.get(
                    "message",
                )

                if isinstance(
                    message,
                    dict,
                ):

                    content = message.get(
                        "content",
                        "",
                    )

                    return str(
                        content or ""
                    ).strip()

                # Defensive fallback.
                content = response.get(
                    "content",
                    "",
                )

                if content:
                    return str(
                        content
                    ).strip()

        except Exception:
            logger.exception(
                "Failed to extract Ollama response content."
            )

        return ""

    # ==================================================================
    # RESPONSE CLEANING
    # ==================================================================

    @staticmethod
    def _clean_response(
        content: str | None,
    ) -> str:
        """
        Clean a complete Ollama response.

        Handles:

        1. <think>reasoning</think>answer
        2. reasoning</think>answer
        3. <think>reasoning
        4. </think>answer
        5. markdown fences
        6. Answer: prefix
        7. Final Answer: prefix
        """

        if content is None:
            return ""

        text = str(
            content
        ).strip()

        if not text:
            return ""

        # --------------------------------------------------------------
        # Remove complete <think>...</think> blocks.
        # --------------------------------------------------------------

        text = re.sub(
            r"<think\b[^>]*>.*?</think\s*>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # --------------------------------------------------------------
        # If an orphan closing tag remains, everything before it is
        # considered reasoning.
        # --------------------------------------------------------------

        closing = re.search(
            r"</think\s*>",
            text,
            flags=re.IGNORECASE,
        )

        if closing:
            text = text[
                closing.end():
            ]

        # --------------------------------------------------------------
        # Remove orphan opening tags.
        # --------------------------------------------------------------

        text = re.sub(
            r"<think\b[^>]*>",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------------------
        # Remove remaining closing tags.
        # --------------------------------------------------------------

        text = re.sub(
            r"</think\s*>",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------------------
        # Remove markdown code fences.
        # --------------------------------------------------------------

        text = re.sub(
            r"^\s*```(?:text|markdown)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------------------
        # Remove common answer headings.
        # --------------------------------------------------------------

        text = re.sub(
            r"^\s*(?:final\s+answer|answer)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------------------
        # Remove accidental surrounding quotes.
        # Only if the entire response is quoted.
        # --------------------------------------------------------------

        if (
            len(text) >= 2
            and text[0] == '"'
            and text[-1] == '"'
        ):
            text = text[1:-1].strip()

        # --------------------------------------------------------------
        # Normalize excessive whitespace.
        # --------------------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ==================================================================
    # OLLAMA OPTIONS
    # ==================================================================

    @staticmethod
    def _options() -> dict[str, Any]:
        """
        Runtime options for local interview generation.

        We intentionally keep generation bounded because interview
        answers should be concise.
        """

        try:
            temperature = float(
                getattr(
                    llm,
                    "temperature",
                    0.2,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            temperature = 0.2

        try:
            max_tokens = int(
                getattr(
                    llm,
                    "max_tokens",
                    256,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            max_tokens = 256

        # Keep answers reasonably short.
        max_tokens = max(
            64,
            min(
                max_tokens,
                384,
            ),
        )

        return {
            "temperature": temperature,
            "num_ctx": 4096,
            "num_predict": max_tokens,
        }

    # ==================================================================
    # GENERATE
    # ==================================================================

    def generate(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Generate one complete response.
        """

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "Prompt must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Generating response | model={}",
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

            # ----------------------------------------------------------
            # Modern Ollama + Qwen3.
            # think=False prevents thinking output when supported.
            # ----------------------------------------------------------

            try:

                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    think=False,
                    options=self._options(),
                )

            except TypeError as exc:

                # ------------------------------------------------------
                # Compatibility with older ollama-python versions that
                # do not support the think argument.
                # ------------------------------------------------------

                if "think" not in str(exc).lower():
                    raise

                logger.warning(
                    "Ollama client does not support think=False. "
                    "Using compatibility mode."
                )

                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    options=self._options(),
                )

        except Exception:

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.exception(
                "Ollama generation failed after {:.2f} sec.",
                elapsed,
            )

            raise

        elapsed = (
            time.perf_counter()
            - start
        )

        raw_content = self._extract_content(
            response
        )

        logger.info(
            "Raw response characters={}",
            len(raw_content),
        )

        final_content = self._clean_response(
            raw_content
        )

        logger.info(
            "Ollama request completed in {:.2f} sec.",
            elapsed,
        )

        logger.info(
            "Final response characters={}",
            len(final_content),
        )

        if not final_content:

            logger.warning(
                "Ollama returned an empty final response."
            )

            return ""

        return final_content

    # ==================================================================
    # STREAM
    # ==================================================================

    def stream(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[str]:
        """
        Stream only final-answer content.

        Thinking tokens are discarded.
        """

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "Prompt must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Streaming response | model={}",
            self.model,
        )

        messages = self._build_messages(
            prompt=prompt,
            history=history,
        )

        start = time.perf_counter()
        total_chars = 0

        try:

            try:

                response_stream = self._client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    think=False,
                    options=self._options(),
                )

            except TypeError as exc:

                if "think" not in str(exc).lower():
                    raise

                logger.warning(
                    "Ollama client does not support think=False "
                    "for streaming."
                )

                response_stream = self._client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    options=self._options(),
                )

            # ----------------------------------------------------------
            # Buffer is used so partial <think> tags do not leak.
            # ----------------------------------------------------------

            buffer = ""

            for chunk in response_stream:

                # ------------------------------------------------------
                # Read chunk safely.
                # ------------------------------------------------------

                token = ""

                try:

                    message = None

                    if isinstance(
                        chunk,
                        dict,
                    ):

                        message = chunk.get(
                            "message",
                            {},
                        )

                        if isinstance(
                            message,
                            dict,
                        ):

                            # NEVER expose thinking.
                            if message.get(
                                "thinking"
                            ):
                                continue

                            token = str(
                                message.get(
                                    "content",
                                    "",
                                )
                                or ""
                            )

                    else:

                        message = getattr(
                            chunk,
                            "message",
                            None,
                        )

                        if message is not None:

                            # NEVER expose thinking.
                            thinking = getattr(
                                message,
                                "thinking",
                                "",
                            )

                            if thinking:
                                continue

                            token = str(
                                getattr(
                                    message,
                                    "content",
                                    "",
                                )
                                or ""
                            )

                except Exception:

                    logger.exception(
                        "Failed to read Ollama stream chunk."
                    )

                    continue

                if not token:
                    continue

                buffer += token

                # ------------------------------------------------------
                # Process complete thinking blocks.
                # ------------------------------------------------------

                while True:

                    opening = re.search(
                        r"<think\b[^>]*>",
                        buffer,
                        flags=re.IGNORECASE,
                    )

                    closing = re.search(
                        r"</think\s*>",
                        buffer,
                        flags=re.IGNORECASE,
                    )

                    # --------------------------------------------------
                    # Opening tag comes first.
                    # --------------------------------------------------

                    if (
                        opening
                        and (
                            not closing
                            or opening.start()
                            < closing.start()
                        )
                    ):

                        # Yield safe text before <think>.
                        prefix = buffer[
                            :opening.start()
                        ]

                        if prefix:

                            total_chars += len(
                                prefix
                            )

                            yield prefix

                        # Keep only text after opening tag.
                        buffer = buffer[
                            opening.end():
                        ]

                        # Wait for </think>.
                        closing_after = re.search(
                            r"</think\s*>",
                            buffer,
                            flags=re.IGNORECASE,
                        )

                        if closing_after:

                            buffer = buffer[
                                closing_after.end():
                            ]

                            continue

                        # Currently inside thinking.
                        buffer = ""

                        break

                    # --------------------------------------------------
                    # Orphan closing tag.
                    # --------------------------------------------------

                    if closing:

                        buffer = buffer[
                            closing.end():
                        ]

                        continue

                    break

                # ------------------------------------------------------
                # If buffer does not contain a possible partial thinking
                # tag, it is safe to emit.
                # ------------------------------------------------------

                if buffer:

                    lower_buffer = buffer.lower()

                    possible_partial_tag = any(
                        tag.startswith(
                            lower_buffer[
                                max(
                                    0,
                                    len(lower_buffer)
                                    - len(tag)
                                ):
                            ]
                        )
                        for tag in (
                            "<think",
                            "</think",
                        )
                    )

                    if not possible_partial_tag:

                        safe_text = buffer

                        buffer = ""

                        cleaned = self._clean_response(
                            safe_text
                        )

                        if cleaned:

                            total_chars += len(
                                cleaned
                            )

                            yield cleaned

            # ----------------------------------------------------------
            # Flush remaining content.
            # ----------------------------------------------------------

            if buffer:

                cleaned = self._clean_response(
                    buffer
                )

                if cleaned:

                    total_chars += len(
                        cleaned
                    )

                    yield cleaned

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Ollama streaming completed in {:.2f} sec.",
                elapsed,
            )

            logger.info(
                "Streamed final-answer characters={}",
                total_chars,
            )

        except Exception:

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.exception(
                "Ollama streaming failed after {:.2f} sec.",
                elapsed,
            )

            raise

    # ==================================================================
    # HEALTH CHECK
    # ==================================================================

    def health_check(self) -> bool:
        """
        Check Ollama availability and model availability.
        """

        logger.info(
            "Running Ollama health check | model={}",
            self.model,
        )

        try:

            # ----------------------------------------------------------
            # Check configured model.
            # ----------------------------------------------------------

            try:

                self._client.show(
                    self.model
                )

            except Exception:

                logger.exception(
                    "Ollama model is unavailable | model={}",
                    self.model,
                )

                return False

            # ----------------------------------------------------------
            # Minimal generation test.
            # ----------------------------------------------------------

            response = self.generate(
                prompt=(
                    "Reply with exactly one word: OK"
                )
            )

            result = response.strip().upper()

            if result == "OK":

                logger.info(
                    "Ollama health check passed."
                )

                return True

            logger.warning(
                "Ollama health check returned {!r}",
                response,
            )

            return False

        except Exception:

            logger.exception(
                "Ollama health check failed."
            )

            return False


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    provider = OllamaProvider()

    print()
    print("=" * 70)
    print("OLLAMA PROVIDER TEST")
    print("=" * 70)

    print()
    print("Model:", provider.model)

    print()
    print("Health check:")

    healthy = provider.health_check()

    print(
        "PASS"
        if healthy
        else "FAIL"
    )

    print()
    print("=" * 70)
    print("TEST 1: EXACT OUTPUT")
    print("=" * 70)

    result = provider.generate(
        "Reply with exactly: TEST"
    )

    print(
        repr(result)
    )

    print()
    print("=" * 70)
    print("TEST 2: INTERVIEW ANSWER")
    print("=" * 70)

    result = provider.generate(
        """
Write a short interview answer.

Question:
Tell me about yourself?

Reference information:
My name is Bharati. I have 1.5 years of experience
in Data Science and Generative AI. I completed my MCA
from Bangalore University in 2024 and my BCA from
Kuvempu University in 2022.

Return only what Bharati should say.
Do not explain anything.
"""
    )

    print(result)

    print()
    print("=" * 70)