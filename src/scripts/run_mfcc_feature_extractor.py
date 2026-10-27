import logging

import yaml

from src.mfcc_feature_extractor import (
    MFCCFeatureExtractor,
)


def load_config(config_path: str) -> dict:
    """Load MFCC extraction configuration from a YAML file."""

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
        r"C:\Users\M S I\PycharmProjects\KWS\configs\mfcc_feature_extraction_config.yaml"
    )

    extraction_config = config[
        "mfcc_extraction"
    ]

    pre_emphasis_config = (
        extraction_config["pre_emphasis"]
    )

    frame_config = extraction_config[
        "frame"
    ]

    window_config = extraction_config[
        "window"
    ]

    fft_config = extraction_config[
        "fft"
    ]

    mel_filter_config = extraction_config[
        "mel_filter"
    ]

    log_config = extraction_config[
        "log"
    ]

    dct_config = extraction_config[
        "dct"
    ]

    mfcc_config = extraction_config[
        "mfcc"
    ]

    extractor = MFCCFeatureExtractor(
        input_directory=extraction_config[
            "input_directory"
        ],
        output_directory=extraction_config[
            "output_directory"
        ],
        sample_rate=extraction_config[
            "sample_rate"
        ],
        duration_seconds=extraction_config[
            "duration_seconds"
        ],
        pre_emphasis_enabled=pre_emphasis_config[
            "enabled"
        ],
        pre_emphasis_coefficient=pre_emphasis_config[
            "coefficient"
        ],
        frame_duration_ms=frame_config[
            "duration_ms"
        ],
        frame_overlap_ratio=frame_config[
            "overlap_ratio"
        ],
        window_type=window_config[
            "type"
        ],
        fft_size=fft_config[
            "size"
        ],
        number_of_mel_filters=mel_filter_config[
            "number_of_filters"
        ],
        minimum_frequency=mel_filter_config[
            "minimum_frequency"
        ],
        maximum_frequency=mel_filter_config[
            "maximum_frequency"
        ],
        log_base=log_config[
            "base"
        ],
        log_epsilon=log_config[
            "epsilon"
        ],
        dct_type=dct_config[
            "type"
        ],
        dct_normalization=dct_config[
            "normalization"
        ],
        number_of_mfcc_coefficients=mfcc_config[
            "number_of_coefficients"
        ],
    )

    extractor.run()


if __name__ == "__main__":
    main()