import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class TrainAudioDataAugmenter:
    """Augment training audio to reduce class imbalance."""

    def __init__(
        self,
        train_directory: str,
        background_noise_directory: str,
        target_sample_rate: int,
        target_duration_seconds: float,
        target_class_count: int,
        noise_enabled: bool,
        snr_levels_db: list[float],
        time_shift_enabled: bool,
        max_shift_seconds: float,
        gain_enabled: bool,
        min_gain: float,
        max_gain: float,
        random_seed: int,
    ) -> None:
        self.train_directory = Path(train_directory)
        self.background_noise_directory = Path(
            background_noise_directory
        )

        self.target_sample_rate = target_sample_rate
        self.target_duration_seconds = (
            target_duration_seconds
        )
        self.target_samples = int(
            target_sample_rate * target_duration_seconds
        )

        self.target_class_count = target_class_count

        self.noise_enabled = noise_enabled
        self.snr_levels_db = snr_levels_db

        self.time_shift_enabled = time_shift_enabled
        self.max_shift_seconds = max_shift_seconds

        self.gain_enabled = gain_enabled
        self.min_gain = min_gain
        self.max_gain = max_gain

        self.random = random.Random(random_seed)

    def find_classes(self) -> list[Path]:
        """Find class directories in the training directory."""

        return sorted(
            directory
            for directory in self.train_directory.iterdir()
            if directory.is_dir()
        )

    def get_audio_files(
        self,
        class_directory: Path,
    ) -> list[Path]:
        """Get all WAV files in a class directory."""

        return sorted(
            class_directory.glob("*.wav")
        )

    def get_original_audio_files(
        self,
        class_directory: Path,
    ) -> list[Path]:
        """Get original WAV files only."""

        audio_files = self.get_audio_files(
            class_directory
        )

        return [
            file
            for file in audio_files
            if "_aug_" not in file.stem
        ]

    def load_audio(
        self,
        file_path: Path,
    ) -> tuple[np.ndarray, int]:
        """Load an audio file."""

        audio, sample_rate = sf.read(
            file_path,
            dtype="float32",
        )

        return audio, sample_rate

    def convert_to_mono(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Convert multi-channel audio to mono."""

        if audio.ndim == 1:
            return audio

        return np.mean(
            audio,
            axis=1,
        )

    def resample_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Resample audio to the target sample rate."""

        if sample_rate == self.target_sample_rate:
            return audio

        greatest_common_divisor = np.gcd(
            sample_rate,
            self.target_sample_rate,
        )

        up = (
            self.target_sample_rate
            // greatest_common_divisor
        )

        down = (
            sample_rate
            // greatest_common_divisor
        )

        return resample_poly(
            audio,
            up,
            down,
        ).astype(np.float32)

    def prepare_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Pad or crop audio to the target duration."""

        if len(audio) < self.target_samples:
            padding_size = (
                self.target_samples - len(audio)
            )

            audio = np.pad(
                audio,
                (0, padding_size),
                mode="constant",
            )

        elif len(audio) > self.target_samples:
            audio = audio[:self.target_samples]

        return audio.astype(np.float32)

    def calculate_rms(
        self,
        audio: np.ndarray,
    ) -> float:
        """Calculate RMS energy."""

        return float(
            np.sqrt(
                np.mean(np.square(audio))
                + 1e-12
            )
        )

    def load_random_background_noise(
        self,
    ) -> np.ndarray:
        """Load a random background-noise recording."""

        noise_files = sorted(
            self.background_noise_directory.glob(
                "*.wav"
            )
        )

        if not noise_files:
            raise FileNotFoundError(
                "No WAV files found in the "
                "background noise directory."
            )

        noise_file = self.random.choice(
            noise_files
        )

        noise, sample_rate = self.load_audio(
            noise_file
        )

        noise = self.convert_to_mono(noise)

        noise = self.resample_audio(
            noise,
            sample_rate,
        )

        return noise

    def get_random_noise_segment(
        self,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Extract a random one-second noise segment."""

        if len(noise) < self.target_samples:
            repeat_count = int(
                np.ceil(
                    self.target_samples
                    / len(noise)
                )
            )

            noise = np.tile(
                noise,
                repeat_count,
            )

        max_start = (
            len(noise)
            - self.target_samples
        )

        start_index = self.random.randint(
            0,
            max_start,
        )

        end_index = (
            start_index
            + self.target_samples
        )

        return noise[
            start_index:end_index
        ]

    def mix_background_noise(
        self,
        speech: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        """Mix background noise with speech."""

        speech_rms = self.calculate_rms(
            speech
        )

        noise_rms = self.calculate_rms(
            noise
        )

        if noise_rms == 0:
            return speech

        desired_noise_rms = (
            speech_rms
            / (10 ** (snr_db / 20))
        )

        noise_scale = (
            desired_noise_rms
            / noise_rms
        )

        mixed_audio = (
            speech
            + noise * noise_scale
        )

        return mixed_audio.astype(
            np.float32
        )

    def apply_time_shift(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Apply a random time shift."""

        max_shift_samples = int(
            self.max_shift_seconds
            * self.target_sample_rate
        )

        shift_samples = self.random.randint(
            -max_shift_samples,
            max_shift_samples,
        )

        if shift_samples == 0:
            return audio

        shifted_audio = np.zeros_like(
            audio
        )

        if shift_samples > 0:
            shifted_audio[
                shift_samples:
            ] = audio[
                :-shift_samples
            ]

        else:
            shift_samples = abs(
                shift_samples
            )

            shifted_audio[
                :-shift_samples
            ] = audio[
                shift_samples:
            ]

        return shifted_audio

    def apply_gain(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Apply a random gain."""

        gain = self.random.uniform(
            self.min_gain,
            self.max_gain,
        )

        return (
            audio * gain
        ).astype(np.float32)

    def normalize_peak(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Keep the waveform within [-1, 1]."""

        peak = np.max(
            np.abs(audio)
        )

        if peak > 1.0:
            audio = audio / peak

        return audio.astype(
            np.float32
        )

    def build_augmented_filename(
        self,
        original_file: Path,
        augmentation_names: list[str],
        sample_index: int,
    ) -> str:
        """Build a filename from applied augmentations."""

        filename_parts = [
            original_file.stem
        ]

        filename_parts.extend(
            augmentation_names
        )

        filename_parts.append(
            f"{sample_index:04d}"
        )

        return (
            "_".join(filename_parts)
            + ".wav"
        )

    def augment_audio(
        self,
        source_file: Path,
    ) -> tuple[np.ndarray, list[str]]:
        """Create one augmented audio sample."""

        audio, sample_rate = self.load_audio(
            source_file
        )

        audio = self.convert_to_mono(
            audio
        )

        audio = self.resample_audio(
            audio,
            sample_rate,
        )

        audio = self.prepare_audio(
            audio
        )

        augmentation_names = []

        if self.noise_enabled:
            noise = (
                self.load_random_background_noise()
            )

            noise = (
                self.get_random_noise_segment(
                    noise
                )
            )

            snr_db = self.random.choice(
                self.snr_levels_db
            )

            audio = self.mix_background_noise(
                audio,
                noise,
                snr_db,
            )

            augmentation_names.append(
                "noise"
            )

        if self.time_shift_enabled:
            audio = self.apply_time_shift(
                audio
            )

            augmentation_names.append(
                "shift"
            )

        if self.gain_enabled:
            audio = self.apply_gain(
                audio
            )

            augmentation_names.append(
                "gain"
            )

        audio = self.normalize_peak(
            audio
        )

        return audio, augmentation_names

    def save_audio(
        self,
        audio: np.ndarray,
        output_file: Path,
    ) -> None:
        """Save audio as a 16-bit PCM WAV file."""

        sf.write(
            output_file,
            audio,
            self.target_sample_rate,
            subtype="PCM_16",
        )

    def augment_class(
        self,
        class_directory: Path,
    ) -> int:
        """Augment one class to the target count."""

        all_audio_files = self.get_audio_files(
            class_directory
        )

        original_files = (
            self.get_original_audio_files(
                class_directory
            )
        )

        current_count = len(
            all_audio_files
        )

        if current_count >= (
            self.target_class_count
        ):
            print(
                f"{class_directory.name}: "
                f"{current_count} files - skip"
            )

            return 0

        if not original_files:
            print(
                f"{class_directory.name}: "
                "no original WAV files - skip"
            )

            return 0

        required_count = (
            self.target_class_count
            - current_count
        )

        print(
            f"{class_directory.name}: "
            f"{current_count} -> "
            f"{self.target_class_count} "
            f"({required_count} new files)"
        )

        generated_count = 0

        while generated_count < required_count:
            source_file = self.random.choice(
                original_files
            )

            audio, augmentation_names = (
                self.augment_audio(
                    source_file
                )
            )

            sample_index = (
                generated_count + 1
            )

            output_filename = (
                self.build_augmented_filename(
                    source_file,
                    augmentation_names,
                    sample_index,
                )
            )

            output_file = (
                class_directory
                / output_filename
            )

            self.save_audio(
                audio,
                output_file,
            )

            generated_count += 1

        return generated_count

    def run(self) -> None:
        """Run augmentation for all training classes."""

        class_directories = (
            self.find_classes()
        )

        total_generated = 0

        for class_directory in class_directories:
            generated_count = (
                self.augment_class(
                    class_directory
                )
            )

            total_generated += generated_count

        print()
        print("=" * 40)
        print("Augmentation Complete")
        print("=" * 40)
        print(
            f"Generated files: "
            f"{total_generated}"
        )
