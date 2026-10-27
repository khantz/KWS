import yaml

from src.audio_validator import AudioValidator


def load_config(config_path: str) -> dict:
    """Load audio validation configuration from a YAML file."""

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config(
        r"C:\Users\M S I\PycharmProjects\KWS\configs\dataset_validation_config.yaml"
    )

    validation_config = config["audio_validation"]

    validator = AudioValidator(
        target_sample_rate=validation_config[
            "target_sample_rate"
        ],
        target_channels=validation_config[
            "target_channels"
        ],
        target_duration_seconds=validation_config[
            "target_duration_seconds"
        ],
        low_rms_threshold=validation_config[
            "low_rms_threshold"
        ],
        clipping_threshold=validation_config[
            "clipping_threshold"
        ],
    )

    results = validator.validate_directory(
        validation_config["input_directory"]
    )

    validator.print_summary(results)
    validator.print_issues(results)

    validator.save_report(
        results,
        validation_config["report_path"],
    )


if __name__ == "__main__":
    main()