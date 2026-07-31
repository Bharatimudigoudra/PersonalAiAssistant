"""
Query Rewriter Service.

Provides a singleton interface for rewriting
follow-up questions into standalone questions.
"""

from app.core.logging import logger
from app.rag.query_rewriter.query_rewriter import (
    QueryRewriter,
)

# ---------------------------------------------------------
# Singleton Query Rewriter
# ---------------------------------------------------------

_query_rewriter = QueryRewriter()


class QueryRewriterService:
    """
    High-level service for query rewriting.
    """

    def __init__(self) -> None:

        self._rewriter = _query_rewriter

        logger.info(
            "QueryRewriterService initialized."
        )

    def rewrite(
        self,
        history: str,
        question: str,
    ) -> str:
        """
        Rewrite a conversational question into
        a standalone question.
        """

        logger.info(
            "Running query rewriter..."
        )

        rewritten = self._rewriter.rewrite(
            history=history,
            question=question,
        )

        logger.info(
            "Query rewriting completed."
        )

        return rewritten


# ---------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------

_query_rewriter_service = QueryRewriterService()


def get_query_rewriter_service() -> QueryRewriterService:
    """
    Return singleton QueryRewriterService.
    """

    return _query_rewriter_service