"""
Conversation memory models.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class ChatMessage:
    """
    Represents one chat message.
    """

    role: str
    content: str