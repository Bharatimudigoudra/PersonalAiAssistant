"""
Prompt Builder.

Builds prompts for the local LLM using retrieved RAG documents.
"""

from __future__ import annotations

from app.core.config import rag
from app.core.logging import logger
from app.models.retrieved_document import RetrievedDocument


class PromptBuilder:
    """
    Builds prompts for interview answering.
    """

    def __init__(self) -> None:

        logger.info(
            "PromptBuilder initialized."
        )

    # ---------------------------------------------------------
    # System Prompt
    # ---------------------------------------------------------

    def build_system_prompt(
        self,
    ) -> str:
        """
        Build the system prompt.
        """

        return (
            "You are Bharati's personal AI interview assistant.\n\n"
            "Your responsibilities:\n"
            "- Answer interview questions professionally.\n"
            "- Use ONLY the provided context.\n"
            "- Never invent experience or projects.\n"
            "- If the answer is unavailable in the context, say:\n"
            "\"I don't have enough information to answer that accurately.\"\n"
            "- Keep answers concise unless asked for more detail.\n"
            "- Answer in first person as Bharati."
        )

    # ---------------------------------------------------------
    # Context
    # ---------------------------------------------------------

    def build_context(
        self,
        documents: list[RetrievedDocument],
    ) -> str:
        """
        Build context from retrieved documents.
        """

        if not documents:

            logger.warning(
                "No retrieved documents."
            )

            return "No context available."

        context_parts = []
        total_chars = 0

        for index, document in enumerate(
            documents,
            start=1,
        ):

            text = document.content.strip()

            if (
                total_chars + len(text)
                > rag.max_context_chars
            ):
                break

            context_parts.append(
                f"[Document {index}]\n{text}"
            )

            total_chars += len(text)

        logger.info(
            "Context built ({} chars).",
            total_chars,
        )

        return "\n\n".join(
            context_parts,
        )

    # ---------------------------------------------------------
    # User Prompt
    # ---------------------------------------------------------

    def build_user_prompt(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> str:
        """
        Build the final user prompt.
        """

        context = self.build_context(
            documents,
        )

        prompt = f"""
CONTEXT
========
{context}

QUESTION
========
{question}

INSTRUCTIONS
============
Answer the interview question using ONLY the context above.

If the context does not contain the answer,
say that you don't have enough information.

Return only the final answer.
"""

        logger.info(
            "User prompt built."
        )

        return prompt.strip()

    # ---------------------------------------------------------
    # Complete Prompt
    # ---------------------------------------------------------

    def build(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> tuple[str, str]:
        """
        Return system prompt and user prompt.
        """

        return (
            self.build_system_prompt(),
            self.build_user_prompt(
                question,
                documents,
            ),
        )


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

_prompt_builder = PromptBuilder()


def get_prompt_builder() -> PromptBuilder:
    """
    Return singleton PromptBuilder.
    """

    return _prompt_builder


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    docs = [
        RetrievedDocument(
            content=(
                "I have 1.5 years of experience in "
                "Generative AI and Data Science."
            ),
            distance=0.1,
        ),
        RetrievedDocument(
            content=(
                "I built RAG systems using "
                "LangChain, ChromaDB, and Ollama."
            ),
            distance=0.2,
        ),
    ]

    builder = get_prompt_builder()

    system_prompt, user_prompt = builder.build(
        question="Tell me about yourself.",
        documents=docs,
    )

    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(system_prompt)

    print()

    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(user_prompt)