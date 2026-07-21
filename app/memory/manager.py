"""
Conversation memory manager.
"""

from collections import deque

from app.core.config import memory
from app.memory.models import ChatMessage


class MemoryManager:
    """
    Stores short-term conversation history.
    """

    def __init__(self):

        self._messages = deque(
            maxlen=memory.max_history
        )

    def add_user_message(
        self,
        content: str,
    ):

        self._messages.append(
            ChatMessage(
                role="user",
                content=content,
            )
        )

    def add_assistant_message(
        self,
        content: str,
    ):

        self._messages.append(
            ChatMessage(
                role="assistant",
                content=content,
            )
        )

    def get_history(self):

        return list(self._messages)

    def clear(self):

        self._messages.clear()