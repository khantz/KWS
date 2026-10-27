import numpy as np


class MelFilterBank:
    """Create and apply a Mel filter bank."""

    def __init__(
        self,
        sample_rate: int = 16000,
        fft_size: int = 512,
        number_of_filters: int = 40,
        minimum_frequency: float = 20.0,
        maximum_frequency: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.number_of_filters = number_of_filters
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency

        self.filters = (
            self.create_filter_bank()
        )

    def apply(
        self,
        power_spectrum: np.ndarray,
    ) -> np.ndarray:
        """Apply the Mel filter bank."""

        return np.matmul(
            power_spectrum,
            self.filters.T,
        ).astype(np.float32)

    def create_filter_bank(
        self,
    ) -> np.ndarray:
        """Create triangular Mel filters."""

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
            self.number_of_filters + 2,
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
                self.number_of_filters,
                number_of_fft_bins,
            ),
            dtype=np.float32,
        )

        for filter_index in range(
            self.number_of_filters
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
        """Convert Hertz to Mel."""

        return (
            2595.0
            * np.log10(
                1.0
                + np.asarray(frequency)
                / 700.0
            )
        )

    @staticmethod
    def mel_to_hz(
        mel: np.ndarray | float,
    ) -> np.ndarray:
        """Convert Mel to Hertz."""

        return (
            700.0
            * (
                10.0 ** (
                    np.asarray(mel)
                    / 2595.0
                )
                - 1.0
            )
        )