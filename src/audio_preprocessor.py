import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocess audio into a fixed format for MFCC extraction."""

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        target_duration_seconds: float = 1.0,
    ):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)

        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_duration_seconds = target_duration_seconds

        self.target_samples = int(
            target_sample_rate * target_duration_seconds
        )

        logger.info(
            "Audio preprocessor initialized: "
            "input=%s, output=%s, sample_rate=%d, "
            "channels=%d, duration=%.2f s, samples=%d",
            self.input_directory,
            self.output_directory,
            self.target_sample_rate,
            self.target_channels,
            self.target_duration_seconds,
            self.target_samples,
        )

    def run(self) -> None:
        """Preprocess all WAV files in the input directory."""

        if not self.input_directory.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{self.input_directory}"
            )

        audio_files = sorted(
            self.input_directory.rglob("*.wav")
        )

        logger.info(
            "Found %d WAV files to preprocess.",
            len(audio_files),
        )

        processed_count = 0
        failed_count = 0

        for input_path in audio_files:
            try:
                self.process_file(input_path)
                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(
                        "Processed %d / %d files.",
                        processed_count,
                        len(audio_files),
                    )

            except Exception:
                failed_count += 1

                logger.exception(
                    "Failed to preprocess: %s",
                    input_path,
                )

        logger.info(
            "Audio preprocessing completed: "
            "processed=%d, failed=%d",
            processed_count,
            failed_count,
        )

    def process_file(
        self,
        input_path: Path,
    ) -> None:
        """Preprocess and save one audio file."""

        audio, sample_rate = sf.read(input_path)

        processed_audio = self.process(
            audio,
            sample_rate,
        )

        output_path = self.get_output_path(
            input_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            output_path,
            processed_audio,
            self.target_sample_rate,
            subtype="FLOAT",
        )

    def get_output_path(
        self,
        input_path: Path,
    ) -> Path:
        """Create the corresponding output path."""

        relative_path = input_path.relative_to(
            self.input_directory
        )

        return self.output_directory / relative_path

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Preprocess one audio waveform.

        Processing order:
        1. Convert to mono.
        2. Resample to target sample rate.
        3. Pad or crop to target duration.
        """

        audio = self.convert_to_mono(audio)

        audio = self.resample(
            audio,
            sample_rate,
        )

        audio = self.pad_or_crop(audio)

        return audio.astype(np.float32)

    def convert_to_mono(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Convert multi-channel audio to mono."""

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim == 1:
            return audio

        if self.target_channels != 1:
            raise ValueError(
                "This preprocessor currently supports "
                "target_channels=1 only."
            )

        logger.debug(
            "Converting %d channels to mono.",
            audio.shape[1],
        )

        return np.mean(
            audio,
            axis=1,
            dtype=np.float32,
        )

    def resample(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Resample audio to the target sample rate."""

        if sample_rate == self.target_sample_rate:
            return audio.astype(np.float32)

        gcd = np.gcd(
            sample_rate,
            self.target_sample_rate,
        )

        up = self.target_sample_rate // gcd
        down = sample_rate // gcd

        logger.debug(
            "Resampling audio: %d Hz -> %d Hz.",
            sample_rate,
            self.target_sample_rate,
        )

        resampled_audio = resample_poly(
            audio,
            up,
            down,
        )

        return resampled_audio.astype(np.float32)

    def pad_or_crop(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Pad short audio or crop long audio to target length."""

        current_samples = len(audio)

        if current_samples == self.target_samples:
            return audio.astype(np.float32)

        if current_samples < self.target_samples:
            padding_samples = (
                self.target_samples - current_samples
            )

            logger.debug(
                "Padding audio: %d -> %d samples.",
                current_samples,
                self.target_samples,
            )

            return np.pad(
                audio,
                (0, padding_samples),
                mode="constant",
            ).astype(np.float32)

        logger.debug(
            "Cropping audio: %d -> %d samples.",
            current_samples,
            self.target_samples,
        )

        return audio[:self.target_samples].astype(
            np.float32
        )