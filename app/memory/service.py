"""
Conversation memory service.
"""

from app.core.logging import logger
from app.memory.manager import MemoryManager
from app.memory.models import ChatMessage


class MemoryService:
    """
    High-level interface for conversation memory.
    """

    def __init__(self) -> None:
        self._memory = MemoryManager()

        logger.info(
            "MemoryService initialized."
        )

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Store a user message.
        """

        self._memory.add_user_message(content)

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        """
        Store an assistant message.
        """

        self._memory.add_assistant_message(content)

    def get_history(
        self,
    ) -> list[ChatMessage]:
        """
        Return the conversation history.
        """

        return self._memory.get_history()

    def clear(
        self,
    ) -> None:
        """
        Clear all conversation history.
        """

        self._memory.clear()

    def size(
        self,
    ) -> int:
        """
        Return the total number of stored messages.
        """

        return self._memory.size()