"""
Ollama LLM Provider.

Local Ollama inference provider for PersonalAiAssistant.

Responsibilities:
- Connect to local Ollama
- Generate normal LLM responses
- Generate RAG responses
- Support streaming
- Disable Qwen thinking when supported
- Never expose model reasoning to the application
- Defensively remove leaked <think> blocks
- Normalize Ollama responses
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
    Concrete LLM provider backed by local Ollama.
    """

    def __init__(self) -> None:
        logger.info("Initializing Ollama Provider...")

        if not llm.base_url:
            raise ValueError(
                "Ollama base URL is not configured."
            )

        if not llm.model_name:
            raise ValueError(
                "Ollama model name is not configured."
            )

        self._client = ollama.Client(
            host=llm.base_url,
        )

        self.model = str(
            llm.model_name
        ).strip()

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
        Global instruction applied to interview/RAG generation.

        Retrieved documents are treated as data, not instructions.
        """

        return """
You are a professional AI interview-answer assistant.

Your job is to produce ONLY the final answer that the candidate should
say to the interviewer.

STRICT RULES:

1. Answer the interviewer's question directly.
2. When the question is about the candidate, answer in first person
   as Bharati.
3. Use only information supplied in the user prompt/context.
4. Never invent experience, education, companies, projects, skills,
   technologies, dates, achievements, responsibilities, or results.
5. Never reveal your reasoning or internal thinking.
6. Never output <think>, </think>, or any hidden reasoning.
7. Never mention RAG, retrieved documents, context, prompts,
   system instructions, LLMs, AI reasoning, or document retrieval.
8. Never say:
   - "We are given the context"
   - "According to the documents"
   - "The context says"
   - "Based on the retrieved documents"
   - "I need to analyze the context"
9. Do not explain how you generated the answer.
10. Do not repeat the interview question.
11. Return only the answer the candidate should speak.
12. Keep answers concise, natural, professional, and interview-ready.
13. For "Tell me about yourself", provide a short professional
    introduction covering education, experience, relevant skills,
    and important projects when those details are available.
14. For behavioral questions, answer naturally in first person using
    only supplied information.
15. If the supplied information is insufficient to answer accurately,
    return exactly:
    I don't have enough information to answer that accurately.
""".strip()

    # ==================================================================
    # MESSAGE BUILDER
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

                try:
                    role = str(
                        item.role
                    ).strip().lower()

                    content = str(
                        item.content
                    ).strip()

                except Exception:
                    logger.warning(
                        "Skipping invalid history message."
                    )
                    continue

                if role not in {
                    "system",
                    "user",
                    "assistant",
                }:
                    role = "user"

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
        Extract final textual content from an Ollama response.

        Ollama Python responses can expose data through objects or
        dictionaries depending on the installed client version.

        We deliberately ignore:
            message.thinking
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
                    "message"
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

                # Some Ollama endpoints may expose
                # response text directly.

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
    # THINKING CLEANUP
    # ==================================================================

    @staticmethod
    def _clean_response(
        content: str | None,
    ) -> str:
        """
        Remove model reasoning from a complete response.

        Handles:

        <think>
        reasoning
        </think>
        final answer

        and:

        reasoning
        </think>
        final answer

        and orphan tags.
        """

        if not content:
            return ""

        text = str(
            content
        ).strip()

        if not text:
            return ""

        # --------------------------------------------------------------
        # Remove complete thinking blocks.
        # --------------------------------------------------------------

        text = re.sub(
            r"<think\b[^>]*>.*?</think\s*>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # --------------------------------------------------------------
        # If a closing tag remains, everything before it is treated
        # as reasoning.
        # --------------------------------------------------------------

        closing_match = re.search(
            r"</think\s*>",
            text,
            flags=re.IGNORECASE,
        )

        if closing_match:

            text = text[
                closing_match.end():
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
        # Remove markdown fences if the model wraps the answer.
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
        # Remove obvious meta-answer prefixes.
        # --------------------------------------------------------------

        text = re.sub(
            r"^\s*(?:final answer|answer)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

    # ==================================================================
    # OLLAMA OPTIONS
    # ==================================================================

    @staticmethod
    def _options() -> dict[str, Any]:
        """
        Runtime options for interview generation.

        The values are intentionally conservative because the application
        is running locally on the user's machine.
        """

        temperature = float(
            getattr(
                llm,
                "temperature",
                0.2,
            )
        )

        max_tokens = int(
            getattr(
                llm,
                "max_tokens",
                256,
            )
        )

        # Interview answers should not become huge.
        max_tokens = max(
            64,
            min(
                max_tokens,
                256,
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
        Generate a complete final answer.
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
            # Qwen3 supports think=False in modern Ollama versions.
            # ----------------------------------------------------------

            response = self._client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                think=False,
                options=self._options(),
            )

        except TypeError as exc:

            # ----------------------------------------------------------
            # Compatibility fallback for older Ollama Python clients
            # that do not accept the `think` argument.
            # ----------------------------------------------------------

            if "think" not in str(exc).lower():

                logger.exception(
                    "Ollama generation failed."
                )

                raise

            logger.warning(
                "Installed Ollama client does not support "
                "think=False. Falling back without think parameter."
            )

            try:

                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    options=self._options(),
                )

            except Exception:

                logger.exception(
                    "Ollama fallback generation failed."
                )

                raise

        except Exception:

            logger.exception(
                "Ollama generation failed."
            )

            raise

        elapsed = (
            time.perf_counter()
            - start
        )

        raw_content = self._extract_content(
            response
        )

        cleaned_content = self._clean_response(
            raw_content
        )

        logger.info(
            "Ollama request completed in {:.2f} sec.",
            elapsed,
        )

        logger.info(
            "Raw response characters={}",
            len(raw_content),
        )

        logger.info(
            "Final response characters={}",
            len(cleaned_content),
        )

        if not cleaned_content:

            logger.warning(
                "Ollama returned an empty final response."
            )

            return ""

        return cleaned_content

    # ==================================================================
    # STREAMING
    # ==================================================================

    def stream(
        self,
        prompt: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[str]:
        """
        Stream only final-answer tokens.

        A small buffer is maintained so <think> tags split across
        network chunks are handled correctly.
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

        # --------------------------------------------------------------
        # Buffer is essential because Ollama may split:
        #
        # <thi
        # nk>
        #
        # across different chunks.
        # --------------------------------------------------------------

        buffer = ""

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

            for chunk in response_stream:

                # ------------------------------------------------------
                # Extract content
                # ------------------------------------------------------

                token = ""

                try:

                    message = getattr(
                        chunk,
                        "message",
                        None,
                    )

                    if message is not None:

                        # Explicitly ignore message.thinking.
                        token = str(
                            getattr(
                                message,
                                "content",
                                "",
                            )
                            or ""
                        )

                    elif isinstance(
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

                            token = str(
                                message.get(
                                    "content",
                                    "",
                                )
                                or ""
                            )

                except Exception:

                    logger.exception(
                        "Failed to process Ollama stream chunk."
                    )

                    continue

                if not token:
                    continue

                # ------------------------------------------------------
                # Add token to buffer.
                # ------------------------------------------------------

                buffer += token

                # ------------------------------------------------------
                # If an opening think tag appears, discard everything
                # until the closing tag.
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
                    # Complete thinking block exists.
                    # --------------------------------------------------

                    if opening and closing:

                        if closing.start() < opening.start():

                            # Orphan closing tag.
                            buffer = buffer[
                                closing.end():
                            ]

                            continue

                        # Remove everything through closing tag.
                        buffer = buffer[
                            closing.end():
                        ]

                        continue

                    # --------------------------------------------------
                    # Opening tag exists without closing tag.
                    # --------------------------------------------------

                    if opening:

                        # Preserve only content before opening tag.
                        prefix = buffer[
                            :opening.start()
                        ]

                        if prefix:

                            total_chars += len(
                                prefix
                            )

                            yield prefix

                        # Keep only the unclosed portion after
                        # opening tag so we can wait for </think>.
                        buffer = buffer[
                            opening.end():
                        ]

                        # Everything currently in this buffer is
                        # thinking until a closing tag arrives.
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
                # We do not immediately yield a buffer that could be a
                # partial <think> tag. Keep a small suffix.
                # ------------------------------------------------------

                if buffer:

                    possible_tag = False

                    for tag in (
                        "<think",
                        "</think",
                    ):

                        for index in range(
                            max(
                                0,
                                len(buffer) - len(tag),
                            ),
                            len(buffer),
                        ):

                            suffix = buffer[
                                index:
                            ].lower()

                            if tag.startswith(
                                suffix
                            ):
                                possible_tag = True
                                break

                        if possible_tag:
                            break

                    if not possible_tag:

                        safe_text = buffer

                        buffer = ""

                        if safe_text:

                            total_chars += len(
                                safe_text
                            )

                            yield safe_text

            # ----------------------------------------------------------
            # Flush remaining safe buffer.
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
        Check whether Ollama is reachable and the configured model
        can generate a minimal response.
        """

        logger.info(
            "Running Ollama health check | model={}",
            self.model,
        )

        try:

            # ----------------------------------------------------------
            # First check Ollama server/model availability.
            # ----------------------------------------------------------

            try:

                self._client.show(
                    self.model
                )

            except Exception:

                logger.exception(
                    "Configured Ollama model is unavailable: {}",
                    self.model,
                )

                return False

            # ----------------------------------------------------------
            # Then perform a tiny generation test.
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
                "Ollama health check returned unexpected "
                "response: {!r}",
                response,
            )

            return False

        except Exception:

            logger.exception(
                "Ollama health check failed."
            )

            return False