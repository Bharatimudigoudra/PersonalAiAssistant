"""
Microphone Recorder.

Records audio from the system microphone and saves it as a WAV file.
"""

from pathlib import Path

import sounddevice as sd
import soundfile as sf

from app.core.logging import logger


class MicrophoneRecorder:
    """
    Records audio from the microphone.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        device: int | None = None,
    ) -> None:

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

        logger.info(
            "Using microphone device {} @ {} Hz ({} channels).",
            self.device,
            self.sample_rate,
            self.channels,
        )

    def record(
        self,
        duration: int,
        output_file: str,
    ) -> Path:
        """
        Record microphone audio.
        """

        logger.info(
            "Recording for {} seconds...",
            duration,
        )

        audio = sd.rec(
            frames=int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )

        sd.wait()

        output_path = Path(output_file).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            file=output_path,
            data=audio,
            samplerate=self.sample_rate,
        )

        logger.info(
            "Recording saved to {}",
            output_path,
        )

        return output_path


def main() -> None:

    recorder = MicrophoneRecorder()

    recorder.record(
        duration=5,
        output_file="app/data/audio/recording.wav",
    )


if __name__ == "__main__":
    main()