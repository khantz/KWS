import numpy as np


class SignalProcessor:
    """Common audio signal-processing operations."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 20.0,
        frame_overlap_ratio: float = 0.30,
        window_type: str = "hamming",
        fft_size: int = 512,
        pre_emphasis_enabled: bool = False,
        pre_emphasis_coefficient: float = 0.97,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_overlap_ratio = frame_overlap_ratio
        self.window_type = window_type
        self.fft_size = fft_size

        self.pre_emphasis_enabled = pre_emphasis_enabled
        self.pre_emphasis_coefficient = (
            pre_emphasis_coefficient
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

    def process(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Convert waveform into a power spectrum."""

        if self.pre_emphasis_enabled:
            audio = self.apply_pre_emphasis(audio)

        frames = self.frame_audio(audio)

        windowed_frames = self.apply_window(
            frames
        )

        power_spectrum = (
            self.calculate_power_spectrum(
                windowed_frames
            )
        )

        return power_spectrum

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
        """Apply the configured window function."""

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
        """Calculate the power spectrum."""

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