"""
Interview Assistant.

Main orchestration service connecting:

    Whisper STT
        ↓
    Hybrid RAG Retrieval
        ↓
    Prompt Builder
        ↓
    Application LLM Service
        ↓
    Ollama Provider
        ↓
    Local LLM

Responsibilities:
- Accept text interview questions
- Accept spoken interview questions
- Retrieve relevant resume/interview information
- Build interview-specific prompts
- Generate personalized answers using local Ollama
"""

from __future__ import annotations

import time

from app.core.logging import logger
from app.speech.whisper_service import get_whisper_service
from app.rag.retrieval import DocumentRetriever
from app.llm.prompt_builder import get_prompt_builder
from app.llm.services import get_llm_service


class InterviewAssistant:
    """
    Main AI Interview Assistant.

    This class orchestrates the complete interview pipeline.
    """

    def __init__(self) -> None:
        logger.info(
            "Initializing InterviewAssistant..."
        )

        # ---------------------------------------------------------
        # Speech-to-Text
        # ---------------------------------------------------------

        self.whisper = get_whisper_service()

        # ---------------------------------------------------------
        # Hybrid RAG Retriever
        # ---------------------------------------------------------

        self.retriever = DocumentRetriever()

        # ---------------------------------------------------------
        # Prompt Builder
        # ---------------------------------------------------------

        self.prompt_builder = get_prompt_builder()

        # ---------------------------------------------------------
        # Application LLM Service
        # ---------------------------------------------------------

        self.llm = get_llm_service()

        logger.info(
            "InterviewAssistant initialized successfully."
        )

    # =============================================================
    # TEXT QUESTION
    # =============================================================

    def answer_question(
        self,
        question: str,
    ) -> str:
        """
        Answer a text-based interview question.

        Pipeline:

            Question
                ↓
            Query rewriting
                ↓
            Vector search
                +
            BM25
                ↓
            RRF
                ↓
            Cross-encoder reranking
                ↓
            Prompt builder
                ↓
            Ollama LLM
        """

        question = question.strip()

        if not question:
            return "Please provide an interview question."

        logger.info(
            "Question: {}",
            question,
        )

        start = time.perf_counter()

        try:

            # -----------------------------------------------------
            # 1. Retrieve relevant documents
            # -----------------------------------------------------

            logger.info(
                "Starting document retrieval..."
            )

            documents = self.retriever.retrieve(
                question
            )

            logger.info(
                "Retrieved {} documents.",
                len(documents),
            )

            # -----------------------------------------------------
            # 2. Build interview prompt
            # -----------------------------------------------------

            logger.info(
                "Building interview prompt..."
            )

            system_prompt, user_prompt = (
                self.prompt_builder.build(
                    question,
                    documents,
                )
            )

            logger.info(
                "Interview prompt built successfully."
            )

            # -----------------------------------------------------
            # 3. Generate answer
            # -----------------------------------------------------

            logger.info(
                "Generating interview answer..."
            )

            answer = self.llm.generate_rag(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            answer = (
                answer.strip()
                if answer
                else ""
            )

            # -----------------------------------------------------
            # 4. Empty-response protection
            # -----------------------------------------------------

            if not answer:

                logger.warning(
                    "LLM returned an empty interview answer."
                )

                return (
                    "I couldn't generate a reliable answer "
                    "from the available interview context."
                )

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Interview pipeline completed in {:.2f} sec.",
                elapsed,
            )

            logger.info(
                "Final answer length: {} characters.",
                len(answer),
            )

            return answer

        except Exception:

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.exception(
                "Interview pipeline failed after {:.2f} sec.",
                elapsed,
            )

            return (
                "I encountered an error while preparing "
                "the interview answer."
            )

    # =============================================================
    # AUDIO QUESTION
    # =============================================================

    def answer_audio(
        self,
        audio_path: str,
    ) -> str:
        """
        Convert spoken interview question to text
        and generate an answer.
        """

        audio_path = str(audio_path).strip()

        if not audio_path:

            return (
                "Audio path cannot be empty."
            )

        logger.info(
            "Processing interview audio: {}",
            audio_path,
        )

        try:

            # -----------------------------------------------------
            # 1. Speech → Text
            # -----------------------------------------------------

            question = self.whisper.transcribe(
                audio_path
            )

            question = (
                question.strip()
                if question
                else ""
            )

            # -----------------------------------------------------
            # 2. Empty transcription protection
            # -----------------------------------------------------

            if not question:

                logger.warning(
                    "Whisper could not recognize the question."
                )

                return (
                    "I could not understand the audio. "
                    "Please try again."
                )

            logger.info(
                "Recognized interview question: {}",
                question,
            )

            # -----------------------------------------------------
            # 3. Send transcription through normal RAG pipeline
            # -----------------------------------------------------

            return self.answer_question(
                question
            )

        except Exception:

            logger.exception(
                "Audio interview pipeline failed."
            )

            return (
                "I encountered an error while "
                "processing the audio."
            )


# ================================================================
# SINGLETON
# ================================================================

_assistant: InterviewAssistant | None = None


def get_interview_assistant() -> InterviewAssistant:
    """
    Return the singleton InterviewAssistant.

    Lazy initialization is used so importing this module does not
    immediately load Whisper, RAG, embeddings, reranker, and LLM.
    """

    global _assistant

    if _assistant is None:

        logger.info(
            "Creating InterviewAssistant singleton."
        )

        _assistant = InterviewAssistant()

    return _assistant


# ================================================================
# COMMAND-LINE TEST
# ================================================================

if __name__ == "__main__":

    assistant = get_interview_assistant()

    print()
    print("=" * 80)
    print("Personal AI Interview Assistant")
    print("=" * 80)
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 80)

    while True:

        try:

            print()

            question = input(
                "Interview Question: "
            ).strip()

        except (KeyboardInterrupt, EOFError):

            print()
            print("Exiting...")
            break

        if not question:

            continue

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Interview Assistant stopped."
            )

            break

        answer = assistant.answer_question(
            question
        )

        print()
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)