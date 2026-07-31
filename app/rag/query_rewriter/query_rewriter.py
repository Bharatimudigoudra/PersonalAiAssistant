"""
Query Rewriter.

Converts follow-up questions into standalone questions
using the configured LLM.
"""

from app.core.logging import logger
from app.llm.services.llm_service import LLMService


class QueryRewriter:
    """
    Rewrites conversational questions into standalone questions.
    """

    def __init__(self) -> None:

        self.llm = LLMService()

        logger.info(
            "QueryRewriter initialized."
        )

    def rewrite(
        self,
        history: str,
        question: str,
    ) -> str:
        """
        Rewrite the user's question into a standalone question.
        """

        logger.info(
            "Rewriting user query..."
        )

        if not history.strip():

            logger.info(
                "No conversation history available."
            )

            return question

        prompt = f"""
You are a query rewriting assistant.

Your task is to convert a follow-up question into a completely standalone question.

Rules:

- Preserve the original meaning.
- Use conversation history.
- Do not answer the question.
- Return ONLY the rewritten question.
- If the question is already standalone,
  return it unchanged.

Conversation History:

{history}

Current Question:

{question}

Standalone Question:
"""

        rewritten = self.llm.generate_rag(
            prompt,
        ).strip()

        if not rewritten:

            logger.warning(
                "Query rewriting failed. Using original question."
            )

            return question

        logger.info(
            "Rewritten question: {}",
            rewritten,
        )

        return rewritten