"""
Memory service.
"""

from app.memory.manager import MemoryManager


class MemoryService:

    def __init__(self):

        self.manager = MemoryManager()

    def save_conversation(
        self,
        question: str,
        answer: str,
    ):

        self.manager.add_user_message(question)

        self.manager.add_assistant_message(answer)

    def history(self):

        return self.manager.get_history()

    def clear(self):

        self.manager.clear()