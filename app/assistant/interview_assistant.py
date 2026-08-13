"""
Personal AI Interview Assistant.

Pipeline:

Question
   ↓
Document Retrieval
   ↓
Prompt Builder
   ↓
LLM Service
   ↓
Interview Answer
"""

from __future__ import annotations

import time

from app.core.logging import logger
from app.llm.prompt_builder import get_prompt_builder
from app.llm.services import get_llm_service
from app.rag.retrieval import DocumentRetriever
from app.speech.whisper_service import get_whisper_service


class InterviewAssistant:
    """
    Main interview assistant orchestration layer.
    """

    def __init__(self) -> None:

        logger.info(
            "Initializing InterviewAssistant..."
        )

        self.whisper = (
            get_whisper_service()
        )

        self.retriever = (
            DocumentRetriever()
        )

        self.prompt_builder = (
            get_prompt_builder()
        )

        self.llm = (
            get_llm_service()
        )

        logger.info(
            "InterviewAssistant initialized successfully."
        )

    # ==============================================================
    # TEXT QUESTION
    # ==============================================================

    def answer_question(
        self,
        question: str,
    ) -> str:

        question = question.strip()

        if not question:

            return (
                "Please provide an interview question."
            )

        start = time.perf_counter()

        logger.info(
            "Question: {}",
            question,
        )

        # ----------------------------------------------------------
        # Retrieval
        # ----------------------------------------------------------

        logger.info(
            "Starting document retrieval..."
        )

        documents = (
            self.retriever.retrieve(
                question
            )
        )

        logger.info(
            "Retrieved {} documents.",
            len(documents),
        )

        # ----------------------------------------------------------
        # Prompt
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Combine system + user prompt
        #
        # generate_rag() accepts ONE prompt.
        # Therefore combine them here.
        # ----------------------------------------------------------

        if system_prompt:

            final_prompt = (
                "SYSTEM INSTRUCTIONS\n"
                "===================\n"
                f"{system_prompt.strip()}\n\n"
                "USER REQUEST\n"
                "============\n"
                f"{user_prompt.strip()}"
            )

        else:

            final_prompt = (
                user_prompt.strip()
            )

        logger.info(
            "Final LLM prompt length={} characters",
            len(final_prompt),
        )

        # ----------------------------------------------------------
        # LLM
        # ----------------------------------------------------------

        logger.info(
            "Generating interview answer..."
        )

        answer = (
            self.llm.generate_rag(
                final_prompt
            )
        )

        answer = answer.strip()

        # ----------------------------------------------------------
        # Empty response handling
        # ----------------------------------------------------------

        if not answer:

            logger.warning(
                "LLM returned an empty interview answer."
            )

            answer = (
                "I couldn't generate a reliable "
                "answer from the available interview context."
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        logger.info(
            "Interview pipeline completed in {:.2f} sec.",
            elapsed,
        )

        return answer

    # ==============================================================
    # AUDIO QUESTION
    # ==============================================================

    def answer_audio(
        self,
        audio_path: str,
    ) -> str:

        logger.info(
            "Processing audio question..."
        )

        question = (
            self.whisper.transcribe(
                audio_path
            )
        )

        question = question.strip()

        if not question:

            return (
                "I could not understand the audio."
            )

        logger.info(
            "Recognized question: {}",
            question,
        )

        return self.answer_question(
            question
        )


# ==============================================================
# SINGLETON
# ==============================================================

_assistant: InterviewAssistant | None = None


def get_interview_assistant() -> InterviewAssistant:

    global _assistant

    if _assistant is None:

        logger.info(
            "Creating InterviewAssistant singleton."
        )

        _assistant = (
            InterviewAssistant()
        )

    return _assistant


# ==============================================================
# CLI
# ==============================================================

def main() -> None:

    assistant = (
        get_interview_assistant()
    )

    print()
    print("=" * 80)
    print("Personal AI Interview Assistant")
    print("=" * 80)
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 80)

    while True:

        try:

            question = input(
                "\nInterview Question: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            print(
                "Exiting Interview Assistant."
            )

            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye."
            )

            break

        answer = (
            assistant.answer_question(
                question
            )
        )

        print()
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)


if __name__ == "__main__":

    main()