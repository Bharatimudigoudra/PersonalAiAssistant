"""
Retrieval-Augmented Generation (RAG) service.
"""

from app.core.logging import logger
from app.llm.prompts.prompt_manager import PromptManager
from app.llm.services.llm_service import LLMService
from app.rag.retrieval import DocumentRetriever


class RAGService:
    """
    Combines document retrieval with LLM generation.
    """

    def __init__(self) -> None:
        self.retriever = DocumentRetriever()
        self.prompt_manager = PromptManager()
        self.llm = LLMService()

    def ask(
        self,
        question: str,
    ) -> str:
        """
        Answer a question using Retrieval-Augmented Generation.
        """

        logger.info("Running RAG pipeline...")

        try:
            # Retrieve relevant chunks
            documents = self.retriever.retrieve(question)

            if not documents:
                logger.warning("No relevant documents found.")
                return "I couldn't find any relevant information."

            # Remove duplicates
            documents = list(dict.fromkeys(documents))

            logger.info(
                "Retrieved {} unique chunks.",
                len(documents),
            )

            # Limit context size (helps prevent Ollama context overflow)
            max_chunks = 3
            documents = documents[:max_chunks]

            context = "\n\n".join(documents)

            logger.info(
                "Context length: {} characters",
                len(context),
            )

            prompt = self.prompt_manager.build_rag_prompt(
                context=context,
                question=question,
            )

            logger.info(
                "Prompt length: {} characters",
                len(prompt),
            )

            # ---------------- DEBUG ----------------
            print("\n" + "=" * 80)
            print("RAG PROMPT")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")
            # ---------------------------------------

            logger.info("Generating response from LLM...")

            response = self.llm.generate(prompt)

            logger.info(
                "LLM returned {} characters.",
                len(response),
            )

            if not response.strip():
                logger.warning("LLM returned an empty response.")

            logger.info("RAG pipeline completed successfully.")

            return response

        except Exception as exc:
            logger.exception(
                "RAG pipeline failed: {}",
                exc,
            )
            return "An error occurred while generating the answer."