"""
Conversation memory models.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Single chat message.
    """

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)