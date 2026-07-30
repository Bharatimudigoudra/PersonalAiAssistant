"""
Retrieval-Augmented Generation (RAG) service.
"""

import time

from app.core.config import rag
from app.core.logging import logger
from app.llm.prompts.prompt_manager import PromptManager
from app.llm.services.llm_service import LLMService
from app.memory.memory_service import get_memory_service
from app.rag.retrieval import (
    DocumentRetriever,
    RetrievedDocument,
)


class RAGService:
    """
    High-level Retrieval-Augmented Generation service.
    """

    def __init__(self) -> None:

        self.retriever = DocumentRetriever()
        self.prompt_manager = PromptManager()
        self.llm = LLMService()
        self.memory = get_memory_service()

        logger.info(
            "RAGService initialized."
        )

    def _build_context(
        self,
        documents: list[RetrievedDocument],
    ) -> str:
        """
        Build context while respecting the configured
        maximum context length.
        """

        context_parts: list[str] = []
        current_length = 0

        for document in documents:

            chunk = document.content.strip()

            if not chunk:
                continue

            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "-",
            )

            context_piece = (
                f"[Source: {source} | Page: {page}]\n"
                f"{chunk}"
            )

            remaining = (
                rag.max_context_chars
                - current_length
            )

            if remaining <= 0:
                break

            if len(context_piece) > remaining:
                context_piece = context_piece[:remaining]

            context_parts.append(
                context_piece
            )

            current_length += len(
                context_piece
            )

        return "\n\n".join(
            context_parts
        )

    def ask(
        self,
        question: str,
    ) -> str:
        """
        Answer a question using Retrieval-Augmented Generation.
        """

        logger.info(
            "Running RAG pipeline..."
        )

        start_time = time.perf_counter()

        try:

            # --------------------------------------------
            # Retrieve documents
            # --------------------------------------------

            documents = self.retriever.retrieve(
                question,
            )

            if not documents:

                logger.warning(
                    "No relevant documents found."
                )

                return (
                    "The information is not available in the provided documents."
                )

            logger.info(
                "Retrieved {} documents.",
                len(documents),
            )

            for index, document in enumerate(
                documents,
                start=1,
            ):

                logger.debug(
                    "[{}] distance={:.4f} rerank={:.4f} source={} page={}",
                    index,
                    document.distance,
                    document.rerank_score,
                    document.metadata.get(
                        "source",
                        "Unknown",
                    ),
                    document.metadata.get(
                        "page",
                        "-",
                    ),
                )

            # --------------------------------------------
            # Build context
            # --------------------------------------------

            context = self._build_context(
                documents,
            )

            if not context:

                logger.warning(
                    "Context is empty."
                )

                return (
                    "The information is not available in the provided documents."
                )

            logger.info(
                "Context length: {} characters.",
                len(context),
            )

            # --------------------------------------------
            # Conversation history
            # --------------------------------------------
                        
            history = self.memory.history_text()

            logger.info(
                "Using {} context chunks.",
                len(documents),
            )

            logger.debug(
                "Conversation history length: {} characters.",
                len(history),
            )

            # --------------------------------------------
            # Prompt
            # --------------------------------------------

            prompt = self.prompt_manager.build_rag_prompt(
                history=history,
                context=context,
                question=question,
            )

            logger.info(
                "Prompt length: {} characters.",
                len(prompt),
            )

            # --------------------------------------------
            # LLM
            # --------------------------------------------

            response = self.llm.generate_rag(
                prompt,
            )

            if not response.strip():

                logger.warning(
                    "LLM returned an empty response."
                )

                return (
                    "The language model returned an empty response."
                )

            logger.info(
                "LLM response length: {} characters.",
                len(response),
            )

            # --------------------------------------------
            # Update conversation memory
            # --------------------------------------------

            self.memory.add_user_message(
                question,
            )

            self.memory.add_assistant_message(
                response,
            )

            logger.info(
                "Conversation memory updated."
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            logger.info(
                "RAG pipeline completed in {:.3f} sec.",
                elapsed,
            )

            return response

        except Exception as exc:

            logger.exception(
                "RAG pipeline failed: {}",
                exc,
            )

            return (
                "An error occurred while generating the answer."
            )