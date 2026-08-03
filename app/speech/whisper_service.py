"""
Whisper Speech-to-Text Service.

Provides a singleton Faster-Whisper model for
offline speech recognition.
"""

from __future__ import annotations

import time
from pathlib import Path

from faster_whisper import WhisperModel

from app.core.config import whisper
from app.core.logging import logger


# ---------------------------------------------------------
# Singleton Whisper Model
# ---------------------------------------------------------

_whisper_model: WhisperModel | None = None


class WhisperService:
    """
    Offline Speech-to-Text service using Faster-Whisper.
    """

    def __init__(self) -> None:

        global _whisper_model

        if _whisper_model is None:

            logger.info(
                "Loading Whisper model: {}",
                whisper.model_name,
            )

            start = time.perf_counter()

            _whisper_model = WhisperModel(
                whisper.model_name,
                device=whisper.device,
                compute_type=whisper.compute_type,
            )

            elapsed = time.perf_counter() - start

            logger.info(
                "Whisper model loaded in {:.2f} sec.",
                elapsed,
            )

        self.model = _whisper_model

    def transcribe(
        self,
        audio_file: str | Path,
    ) -> str:
        """
        Convert an audio file into text.
        """

        audio_file = Path(audio_file)

        if not audio_file.exists():

            logger.error(
                "Audio file not found: {}",
                audio_file,
            )

            return ""

        logger.info(
            "Transcribing: {}",
            audio_file,
        )

        start = time.perf_counter()

        try:

            segments, info = self.model.transcribe(
                str(audio_file),
                beam_size=5,
                vad_filter=True,
                language=None,
            )

            transcript_parts: list[str] = []

            for segment in segments:

                text = segment.text.strip()

                if text:

                    transcript_parts.append(
                        text,
                    )

            transcript = " ".join(
                transcript_parts,
            ).strip()

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Detected language: {}",
                info.language,
            )

            logger.info(
                "Transcription completed in {:.2f} sec.",
                elapsed,
            )

            logger.info(
                "Transcript length: {} characters.",
                len(transcript),
            )

            if not transcript:

                logger.warning(
                    "Whisper returned an empty transcript."
                )

            return transcript

        except Exception as exc:

            logger.exception(
                "Whisper transcription failed: {}",
                exc,
            )

            return ""


# ---------------------------------------------------------
# Singleton Service
# ---------------------------------------------------------

_whisper_service = WhisperService()


def get_whisper_service() -> WhisperService:
    """
    Return singleton WhisperService.
    """

    return _whisper_service


# ---------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------

if __name__ == "__main__":

    service = get_whisper_service()

    audio_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "data"
        / "audio"
        / "recording.wav"
    )

    result = service.transcribe(
        audio_path,
    )

    print()
    print("=" * 60)
    print(result)
    print("=" * 60)