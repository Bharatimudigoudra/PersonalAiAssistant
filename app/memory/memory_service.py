"""
Conversation memory service.

Maintains conversation history for the assistant.
"""

from collections import deque

from app.core.config import memory
from app.core.logging import logger
from app.memory.memory import ChatMessage


class MemoryService:
    """
    Stores conversation history in memory.
    """

    def __init__(self) -> None:

        self._messages = deque(
            maxlen=memory.max_history,
        )

        logger.info(
            "MemoryService initialized (max history={}).",
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

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """
        Store a message with any role.
        """

        self._messages.append(
            ChatMessage(
                role=role,
                content=content,
            )
        )

    def history(self) -> list[ChatMessage]:
        """
        Return conversation history.
        """

        return list(self._messages)

    def history_text(self) -> str:
        """
        Return conversation history formatted for prompts.
        """

        if not self._messages:
            return ""

        return "\n".join(
            f"{message.role.title()}: {message.content}"
            for message in self._messages
        )

    def clear(self) -> None:
        """
        Clear all stored messages.
        """

        self._messages.clear()

        logger.info(
            "Conversation memory cleared."
        )

    def size(self) -> int:
        """
        Return number of stored messages.
        """

        return len(self._messages)


# ---------------------------------------------------------------------
# Singleton Memory Service
# ---------------------------------------------------------------------

_memory_service = MemoryService()


def get_memory_service() -> MemoryService:
    """
    Return the shared MemoryService instance.
    """

    return _memory_service