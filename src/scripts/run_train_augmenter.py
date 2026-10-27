import yaml

from src.train_audio_data_argumenter import (
    TrainAudioDataAugmenter,
)


def load_config(config_path: str) -> dict:
    """Load augmentation configuration from a YAML file."""

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config(
        r"C:\Users\M S I\PycharmProjects\KWS\configs\train_dataset_augmetation_config.yaml"
    )

    augmentation_config = config["audio_augmentation"]

    background_noise_config = (
        augmentation_config["background_noise"]
    )

    time_shift_config = (
        augmentation_config["time_shift"]
    )

    gain_config = (
        augmentation_config["gain"]
    )

    augmenter = TrainAudioDataAugmenter(
        train_directory=augmentation_config[
            "train_directory"
        ],
        background_noise_directory=augmentation_config[
            "background_noise_directory"
        ],
        target_sample_rate=augmentation_config[
            "target_sample_rate"
        ],
        target_duration_seconds=augmentation_config[
            "target_duration_seconds"
        ],
        target_class_count=augmentation_config[
            "target_class_count"
        ],
        noise_enabled=background_noise_config[
            "enabled"
        ],
        snr_levels_db=background_noise_config[
            "snr_levels_db"
        ],
        time_shift_enabled=time_shift_config[
            "enabled"
        ],
        max_shift_seconds=time_shift_config[
            "max_shift_seconds"
        ],
        gain_enabled=gain_config[
            "enabled"
        ],
        min_gain=gain_config[
            "min_gain"
        ],
        max_gain=gain_config[
            "max_gain"
        ],
        random_seed=augmentation_config[
            "random_seed"
        ],
    )

    augmenter.run()


if __name__ == "__main__":
    main()