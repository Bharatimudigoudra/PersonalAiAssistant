"""
Prompt manager.
"""

from app.llm.prompts.base import SYSTEM_HEADER
from app.llm.prompts.interview import INTERVIEW_PROMPT
from app.llm.prompts.rag import RAG_PROMPT
from app.llm.prompts.system import SYSTEM_PROMPT


class PromptManager:
    """
    Builds prompts for different LLM tasks.
    """

    def build_system_prompt(
        self,
    ) -> str:
        """
        Build the shared system prompt.
        """

        return "\n\n".join(
            [
                SYSTEM_HEADER,
                SYSTEM_PROMPT,
            ]
        )

    def build_interview_prompt(
        self,
        question: str,
    ) -> str:
        """
        Build an interview prompt.
        """

        return "\n\n".join(
            [
                self.build_system_prompt(),
                INTERVIEW_PROMPT.format(
                    question=question,
                ),
            ]
        )

    def build_rag_prompt(
        self,
        history: str,
        context: str,
        question: str,
    ) -> str:
        """
        Build a RAG prompt.
        """

        return "\n\n".join(
            [
                self.build_system_prompt(),
                RAG_PROMPT.format(
                    history=history,
                    context=context,
                    question=question,
                ),
            ]
        )