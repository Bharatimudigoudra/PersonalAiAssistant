"""
Interview Assistant.

Main orchestration service that connects

- Whisper STT
- Hybrid RAG Retrieval
- Prompt Builder
- Local LLM

into one pipeline.
"""

from __future__ import annotations

import time

from app.core.logging import logger

from app.speech.whisper_service import (
    get_whisper_service,
)

from app.rag.retrieval import (
    DocumentRetriever,
)

from app.llm.prompt_builder import (
    get_prompt_builder,
)

from app.llm.llm_service import (
    get_llm_service,
)


class InterviewAssistant:
    """
    Main AI Interview Assistant.
    """

    def __init__(self) -> None:

        self.whisper = get_whisper_service()

        self.retriever = DocumentRetriever()

        self.prompt_builder = get_prompt_builder()

        self.llm = get_llm_service()

        logger.info(
            "InterviewAssistant initialized."
        )

    # -----------------------------------------------------

    def answer_question(
        self,
        question: str,
    ) -> str:
        """
        Answer a text question.
        """

        logger.info(
            "Question: {}",
            question,
        )

        start = time.perf_counter()

        documents = self.retriever.retrieve(
            question,
        )

        system_prompt, user_prompt = (
            self.prompt_builder.build(
                question,
                documents,
            )
        )

        answer = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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

    # -----------------------------------------------------

    def answer_audio(
        self,
        audio_path: str,
    ) -> str:
        """
        Answer from spoken audio.
        """

        logger.info(
            "Processing audio..."
        )

        question = self.whisper.transcribe(
            audio_path,
        )

        if not question:

            return "I could not understand the audio."

        logger.info(
            "Recognized question: {}",
            question,
        )

        return self.answer_question(
            question,
        )


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

_assistant = InterviewAssistant()


def get_interview_assistant() -> InterviewAssistant:
    """
    Return singleton assistant.
    """

    return _assistant


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    assistant = get_interview_assistant()

    while True:

        print()

        question = input(
            "Interview Question (exit to quit): "
        )

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        answer = assistant.answer_question(
            question,
        )

        print()

        print("=" * 80)

        print(answer)

        print("=" * 80)