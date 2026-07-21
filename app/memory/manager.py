"""
Conversation memory manager.

Maintains short-term conversation history for the assistant.
"""

from collections import deque

from app.core.config import memory
from app.core.logging import logger
from app.memory.models import ChatMessage


class MemoryManager:
    """
    Stores short-term conversation history.
    """

    def __init__(self) -> None:
        """
        Initialize memory manager.
        """

        self._messages: deque[ChatMessage] = deque(
            maxlen=memory.max_history,
        )

        logger.info(
            "MemoryManager initialized (max_history={})",
            memory.max_history,
        )

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Store a user message.
        """

        self._messages.append(
            ChatMessage(
                role="user",
                content=content,
            )
        )

        logger.debug("User message added.")

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        """
        Store an assistant message.
        """

        self._messages.append(
            ChatMessage(
                role="assistant",
                content=content,
            )
        )

        logger.debug("Assistant message added.")

    def get_history(
        self,
    ) -> list[ChatMessage]:
        """
        Return the conversation history.
        """

        return list(self._messages)

    def clear(
        self,
    ) -> None:
        """
        Clear conversation history.
        """

        self._messages.clear()

        logger.info("Conversation memory cleared.")

    def size(
        self,
    ) -> int:
        """
        Return the number of stored messages.
        """

        return len(self._messages)