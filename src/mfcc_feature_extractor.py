import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct


logger = logging.getLogger(__name__)


class MFCCFeatureExtractor:
    """Extract MFCC features from preprocessed WAV files."""

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        sample_rate: int = 16000,
        duration_seconds: float = 1.0,
        pre_emphasis_enabled: bool = True,
        pre_emphasis_coefficient: float = 0.97,
        frame_duration_ms: float = 20.0,
        frame_overlap_ratio: float = 0.30,
        window_type: str = "hamming",
        fft_size: int = 512,
        number_of_mel_filters: int = 26,
        minimum_frequency: float = 20.0,
        maximum_frequency: float = 8000.0,
        log_base: str = "natural",
        log_epsilon: float = 1.0e-10,
        dct_type: int = 2,
        dct_normalization: str = "ortho",
        number_of_mfcc_coefficients: int = 13,
    ):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)

        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds

        self.pre_emphasis_enabled = (
            pre_emphasis_enabled
        )
        self.pre_emphasis_coefficient = (
            pre_emphasis_coefficient
        )

        self.frame_duration_ms = frame_duration_ms
        self.frame_overlap_ratio = (
            frame_overlap_ratio
        )

        self.window_type = window_type
        self.fft_size = fft_size

        self.number_of_mel_filters = (
            number_of_mel_filters
        )
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency

        self.log_base = log_base
        self.log_epsilon = log_epsilon

        self.dct_type = dct_type
        self.dct_normalization = dct_normalization
        self.number_of_mfcc_coefficients = (
            number_of_mfcc_coefficients
        )

        self.frame_length = int(
            sample_rate
            * frame_duration_ms
            / 1000
        )

        self.frame_overlap = int(
            self.frame_length
            * frame_overlap_ratio
        )

        self.hop_length = (
            self.frame_length
            - self.frame_overlap
        )

        self.expected_samples = int(
            sample_rate * duration_seconds
        )

        self.mel_filter_bank = (
            self.create_mel_filter_bank()
        )

        logger.info(
            "MFCC extractor initialized: "
            "sample_rate=%d, frame_length=%d, "
            "hop_length=%d, fft_size=%d, "
            "mel_filters=%d, mfcc_coefficients=%d",
            self.sample_rate,
            self.frame_length,
            self.hop_length,
            self.fft_size,
            self.number_of_mel_filters,
            self.number_of_mfcc_coefficients,
        )

    def run(self) -> None:
        """Extract MFCC features from all WAV files."""

        if not self.input_directory.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{self.input_directory}"
            )

        audio_files = sorted(
            self.input_directory.rglob("*.wav")
        )

        logger.info(
            "Found %d WAV files for MFCC extraction.",
            len(audio_files),
        )

        extracted_count = 0
        failed_count = 0

        for input_path in audio_files:
            try:
                self.process_file(input_path)

                extracted_count += 1

                if extracted_count % 100 == 0:
                    logger.info(
                        "Extracted MFCC: %d / %d files.",
                        extracted_count,
                        len(audio_files),
                    )

            except Exception:
                failed_count += 1

                logger.exception(
                    "Failed to extract MFCC: %s",
                    input_path,
                )

        logger.info(
            "MFCC extraction completed: "
            "extracted=%d, failed=%d",
            extracted_count,
            failed_count,
        )

    def process_file(
        self,
        input_path: Path,
    ) -> None:
        """Extract and save MFCC features for one WAV file."""

        audio, sample_rate = sf.read(
            input_path,
            dtype="float32",
        )

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample rate "
                f"{self.sample_rate} Hz, "
                f"got {sample_rate} Hz."
            )

        if audio.ndim != 1:
            raise ValueError(
                f"Expected mono audio, "
                f"got shape {audio.shape}."
            )

        if len(audio) != self.expected_samples:
            raise ValueError(
                f"Expected {self.expected_samples} "
                f"samples, got {len(audio)}."
            )

        mfcc_features = self.extract(audio)

        output_path = self.get_output_path(
            input_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            mfcc_features,
        )

    def get_output_path(
        self,
        input_path: Path,
    ) -> Path:
        """Create the corresponding output feature path."""

        relative_path = input_path.relative_to(
            self.input_directory
        )

        return (
            self.output_directory
            / relative_path.with_suffix(".npy")
        )

    def extract(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Extract MFCC features from one waveform."""

        if self.pre_emphasis_enabled:
            audio = self.apply_pre_emphasis(audio)

        frames = self.frame_audio(audio)

        windowed_frames = self.apply_window(
            frames
        )

        power_spectrum = self.calculate_power_spectrum(
            windowed_frames
        )

        mel_energies = self.apply_mel_filter_bank(
            power_spectrum
        )

        log_mel_energies = self.apply_log(
            mel_energies
        )

        mfcc_features = self.apply_dct(
            log_mel_energies
        )

        return mfcc_features.astype(
            np.float32
        )

    def apply_pre_emphasis(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Apply first-order pre-emphasis."""

        emphasized_audio = np.empty_like(
            audio,
            dtype=np.float32,
        )

        emphasized_audio[0] = audio[0]

        emphasized_audio[1:] = (
            audio[1:]
            - self.pre_emphasis_coefficient
            * audio[:-1]
        )

        return emphasized_audio

    def frame_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Split audio into overlapping frames."""

        number_of_frames = (
            1
            + (
                len(audio) - self.frame_length
            )
            // self.hop_length
        )

        if number_of_frames <= 0:
            raise ValueError(
                "Audio is shorter than one frame."
            )

        frames = np.empty(
            (
                number_of_frames,
                self.frame_length,
            ),
            dtype=np.float32,
        )

        for frame_index in range(
            number_of_frames
        ):
            start = (
                frame_index
                * self.hop_length
            )

            end = (
                start
                + self.frame_length
            )

            frames[frame_index] = audio[
                start:end
            ]

        return frames

    def apply_window(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:
        """Apply the selected window function."""

        if self.window_type.lower() != "hamming":
            raise ValueError(
                f"Unsupported window type: "
                f"{self.window_type}"
            )

        window = np.hamming(
            self.frame_length
        ).astype(np.float32)

        return frames * window

    def calculate_power_spectrum(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:
        """Calculate the power spectrum of each frame."""

        fft_result = np.fft.rfft(
            frames,
            n=self.fft_size,
            axis=1,
        )

        power_spectrum = (
            np.abs(fft_result) ** 2
        ) / self.fft_size

        return power_spectrum.astype(
            np.float32
        )

    def apply_mel_filter_bank(
        self,
        power_spectrum: np.ndarray,
    ) -> np.ndarray:
        """Apply the Mel filterbank to the power spectrum."""

        return np.matmul(
            power_spectrum,
            self.mel_filter_bank.T,
        ).astype(np.float32)

    def apply_log(
        self,
        mel_energies: np.ndarray,
    ) -> np.ndarray:
        """Apply the configured logarithm."""

        if self.log_base.lower() != "natural":
            raise ValueError(
                f"Unsupported log base: "
                f"{self.log_base}"
            )

        return np.log(
            mel_energies + self.log_epsilon
        ).astype(np.float32)

    def apply_dct(
        self,
        log_mel_energies: np.ndarray,
    ) -> np.ndarray:
        """Apply DCT and keep the requested MFCC coefficients."""

        mfcc = dct(
            log_mel_energies,
            type=self.dct_type,
            axis=1,
            norm=self.dct_normalization,
        )

        return mfcc[
            :,
            :self.number_of_mfcc_coefficients,
        ]

    def create_mel_filter_bank(
        self,
    ) -> np.ndarray:
        """Create triangular Mel filterbank."""

        if (
            self.minimum_frequency < 0
            or self.maximum_frequency
            > self.sample_rate / 2
        ):
            raise ValueError(
                "Mel frequency range must be "
                "within the Nyquist frequency."
            )

        if (
            self.minimum_frequency
            >= self.maximum_frequency
        ):
            raise ValueError(
                "Minimum frequency must be lower "
                "than maximum frequency."
            )

        mel_min = self.hz_to_mel(
            self.minimum_frequency
        )

        mel_max = self.hz_to_mel(
            self.maximum_frequency
        )

        mel_points = np.linspace(
            mel_min,
            mel_max,
            self.number_of_mel_filters + 2,
        )

        hz_points = self.mel_to_hz(
            mel_points
        )

        frequency_bins = np.floor(
            (
                self.fft_size + 1
            )
            * hz_points
            / self.sample_rate
        ).astype(int)

        number_of_fft_bins = (
            self.fft_size // 2 + 1
        )

        filter_bank = np.zeros(
            (
                self.number_of_mel_filters,
                number_of_fft_bins,
            ),
            dtype=np.float32,
        )

        for filter_index in range(
            self.number_of_mel_filters
        ):
            left = frequency_bins[
                filter_index
            ]

            center = frequency_bins[
                filter_index + 1
            ]

            right = frequency_bins[
                filter_index + 2
            ]

            if center > left:
                for bin_index in range(
                    left,
                    center,
                ):
                    filter_bank[
                        filter_index,
                        bin_index,
                    ] = (
                        bin_index - left
                    ) / (
                        center - left
                    )

            if right > center:
                for bin_index in range(
                    center,
                    right,
                ):
                    filter_bank[
                        filter_index,
                        bin_index,
                    ] = (
                        right - bin_index
                    ) / (
                        right - center
                    )

        return filter_bank

    @staticmethod
    def hz_to_mel(
        frequency: np.ndarray | float,
    ) -> np.ndarray:
        """Convert frequency from Hertz to Mel."""

        return (
            2595.0
            * np.log10(
                1.0 + np.asarray(frequency) / 700.0
            )
        )

    @staticmethod
    def mel_to_hz(
        mel: np.ndarray | float,
    ) -> np.ndarray:
        """Convert frequency from Mel to Hertz."""

        return (
            700.0
            * (
                10.0 ** (
                    np.asarray(mel) / 2595.0
                )
                - 1.0
            )
        )