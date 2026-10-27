import logging

import yaml

from src.audio_preprocessor import AudioPreprocessor


def load_config(config_path: str) -> dict:
    """Load audio preprocessing configuration from a YAML file."""

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = load_config(
        r"C:\Users\M S I\PycharmProjects\KWS\configs\audio_prerocessing_config.yaml"
    )

    preprocessing_config = config[
        "audio_preprocessing"
    ]

    preprocessor = AudioPreprocessor(
        input_directory=preprocessing_config[
            "input_directory"
        ],
        output_directory=preprocessing_config[
            "output_directory"
        ],
        target_sample_rate=preprocessing_config[
            "target_sample_rate"
        ],
        target_channels=preprocessing_config[
            "target_channels"
        ],
        target_duration_seconds=preprocessing_config[
            "target_duration_seconds"
        ],
    )

    preprocessor.run()


if __name__ == "__main__":
    main()