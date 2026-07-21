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
    SystemMessage,
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
            num_predict=1024,
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

        # --------------------------------------------------
        # Build conversation
        # --------------------------------------------------

        messages = [
            SystemMessage(
                content="""
You are Personal AI Assistant.

Answer directly.

Do not reveal your reasoning.

Return only the final answer.

Be concise unless the user asks for more detail.
""".strip()
            )
        ]

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

            logger.info("No conversation history found.")

            messages.append(
                HumanMessage(
                    content=prompt,
                )
            )

        logger.info(
            "Sending {} messages to Ollama.",
            len(messages),
        )

        # --------------------------------------------------
        # Generate response
        # --------------------------------------------------

        response = self._llm.invoke(messages)

        elapsed = time.time() - start

        logger.info(
            "LLM completed in {:.2f} seconds.",
            elapsed,
        )

        logger.info(
            "Response type: {}",
            type(response).__name__,
        )

        logger.info(
            "Response content length: {}",
            len(response.content or ""),
        )

        if not response.content:

            logger.warning(
                "LLM returned an empty response."
            )

            logger.debug(
                "Response metadata: {}",
                response.response_metadata,
            )

            return ""

        return response.content.strip()

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        """
        Stream the model response.
        """

        logger.info("Streaming response...")

        start = time.time()

        messages = [
            SystemMessage(
                content="""
You are Personal AI Assistant.

Answer directly.

Do not reveal your reasoning.

Return only the final answer.
""".strip()
            ),
            HumanMessage(
                content=prompt,
            ),
        ]

        for chunk in self._llm.stream(messages):

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

            self.generate(
                prompt="Hello",
                history=None,
            )

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