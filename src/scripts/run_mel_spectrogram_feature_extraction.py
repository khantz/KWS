import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from src.feature_extraction.mel_spectrogram_extractor import (
    MelSpectrogramFeatureExtractor,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\configs\mel_spectrogram_feature_extraction_config.yaml"
)


# ============================================================
# Configuration Loader
# ============================================================

def load_config(
    config_path: Path,
) -> dict:

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# Main Pipeline
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config(
        CONFIG_PATH
    )

    input_directory = Path(
        config["dataset"]["input_directory"]
    )

    output_directory = Path(
        config["dataset"]["output_directory"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Feature configuration
    # --------------------------------------------------------

    feature_config = config[
        "mel_spectrogram"
    ]

    sample_rate = feature_config[
        "sample_rate"
    ]

    duration_seconds = feature_config[
        "duration_seconds"
    ]

    expected_samples = int(
        sample_rate
        * duration_seconds
    )

    # --------------------------------------------------------
    # Create extractor
    # --------------------------------------------------------

    extractor = MelSpectrogramFeatureExtractor(
        sample_rate=sample_rate,

        pre_emphasis_enabled=feature_config[
            "pre_emphasis"
        ]["enabled"],

        pre_emphasis_coefficient=feature_config[
            "pre_emphasis"
        ]["coefficient"],

        frame_duration_ms=feature_config[
            "frame"
        ]["duration_ms"],

        frame_overlap_ratio=feature_config[
            "frame"
        ]["overlap_ratio"],

        window_type=feature_config[
            "window"
        ]["type"],

        fft_size=feature_config[
            "fft"
        ]["size"],

        number_of_mel_filters=feature_config[
            "mel_filter"
        ]["number_of_filters"],

        minimum_frequency=feature_config[
            "mel_filter"
        ]["minimum_frequency"],

        maximum_frequency=feature_config[
            "mel_filter"
        ]["maximum_frequency"],

        log_epsilon=feature_config[
            "log"
        ]["epsilon"],
    )

    # --------------------------------------------------------
    # Find WAV files
    # --------------------------------------------------------

    if not input_directory.exists():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_directory}"
        )

    audio_files = sorted(
        input_directory.rglob("*.wav")
    )

    logger.info(
        "Found %d WAV files.",
        len(audio_files),
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    extracted_count = 0
    failed_count = 0

    for input_path in audio_files:

        try:

            # ----------------------------------------------
            # Read audio
            # ----------------------------------------------

            audio, actual_sample_rate = (
                sf.read(
                    input_path,
                    dtype="float32",
                )
            )

            # ----------------------------------------------
            # Validate sample rate
            # ----------------------------------------------

            if actual_sample_rate != sample_rate:

                raise ValueError(
                    f"Expected sample rate "
                    f"{sample_rate} Hz, "
                    f"got "
                    f"{actual_sample_rate} Hz."
                )

            # ----------------------------------------------
            # Validate mono
            # ----------------------------------------------

            if audio.ndim != 1:

                raise ValueError(
                    f"Expected mono audio, "
                    f"got shape {audio.shape}."
                )

            # ----------------------------------------------
            # Validate duration
            # ----------------------------------------------

            if len(audio) != expected_samples:

                raise ValueError(
                    f"Expected "
                    f"{expected_samples} samples, "
                    f"got {len(audio)}."
                )

            # ----------------------------------------------
            # Extract Mel spectrogram
            # ----------------------------------------------

            features = extractor.extract(
                audio
            )

            # ----------------------------------------------
            # Create output path
            # ----------------------------------------------

            relative_path = (
                input_path.relative_to(
                    input_directory
                )
            )

            output_path = (
                output_directory
                / relative_path.with_suffix(
                    ".npy"
                )
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------
            # Save feature
            # ----------------------------------------------

            np.save(
                output_path,
                features,
            )

            extracted_count += 1

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            if extracted_count % 100 == 0:

                logger.info(
                    "Extracted Mel spectrogram: "
                    "%d / %d files.",
                    extracted_count,
                    len(audio_files),
                )

        except Exception:

            failed_count += 1

            logger.exception(
                "Failed to process: %s",
                input_path,
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    logger.info(
        "Mel-spectrogram extraction completed: "
        "extracted=%d, failed=%d",
        extracted_count,
        failed_count,
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()