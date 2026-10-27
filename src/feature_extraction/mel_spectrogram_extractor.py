import numpy as np

from src.feature_extraction.signal_processor import (
    SignalProcessor,
)
from src.feature_extraction.mel_filter_bank import (
    MelFilterBank,
)


class MelSpectrogramFeatureExtractor:
    """Extract log-Mel spectrogram features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        pre_emphasis_enabled: bool = False,
        pre_emphasis_coefficient: float = 0.97,
        frame_duration_ms: float = 20.0,
        frame_overlap_ratio: float = 0.30,
        window_type: str = "hamming",
        fft_size: int = 512,
        number_of_mel_filters: int = 40,
        minimum_frequency: float = 20.0,
        maximum_frequency: float = 8000.0,
        log_epsilon: float = 1.0e-10,
    ):
        self.signal_processor = SignalProcessor(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            frame_overlap_ratio=frame_overlap_ratio,
            window_type=window_type,
            fft_size=fft_size,
            pre_emphasis_enabled=(
                pre_emphasis_enabled
            ),
            pre_emphasis_coefficient=(
                pre_emphasis_coefficient
            ),
        )

        self.mel_filter_bank = MelFilterBank(
            sample_rate=sample_rate,
            fft_size=fft_size,
            number_of_filters=(
                number_of_mel_filters
            ),
            minimum_frequency=(
                minimum_frequency
            ),
            maximum_frequency=(
                maximum_frequency
            ),
        )

        self.log_epsilon = log_epsilon

    def extract(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Extract a log-Mel spectrogram."""

        power_spectrum = (
            self.signal_processor.process(
                audio
            )
        )

        mel_energies = (
            self.mel_filter_bank.apply(
                power_spectrum
            )
        )

        log_mel = np.log(
            mel_energies
            + self.log_epsilon
        ).astype(np.float32)

        return log_mel