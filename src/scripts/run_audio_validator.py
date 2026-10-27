from pathlib import Path

import yaml

from src.audio_validator import AudioValidator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

config_path = (
    PROJECT_ROOT
    / "configs"
    / "dataset_validation_config.yaml"
)


with config_path.open(
    "r",
    encoding="utf-8",
) as config_file:
    config = yaml.safe_load(config_file)


validation_config = config["audio_validation"]

input_directory = Path(
    validation_config["input_directory"]
)

report_path = Path(
    validation_config["report_path"]
)


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
    input_directory
)

validator.print_summary(results)
validator.print_issues(results)

validator.save_report(
    results,
    report_path,
)