
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from src.feature_loader import MFCCFeatureLoader
from src.feature_normalizer import FeatureNormalizer
from src.tf_lite_converter import TFLiteConverter


PROJECT_DIR = Path(r"C:\Users\M S I\PycharmProjects\KWS")
MODELS_DIR = PROJECT_DIR / "Models"

MODELS = [
    {
        "name": "CNN1D",
        "config": PROJECT_DIR / "configs" / "oned_cnn_training_config.yaml",
        "keras_model": MODELS_DIR / "CNN1D_keyword_spotting.keras",
        "output": MODELS_DIR / "CNN1D_keyword_spotting_int8.tflite",
        "add_channel": False,
    },
    {
        "name": "DS-CNN",
        "config": PROJECT_DIR / "configs" / "dscnn_training_config.yaml",
        "keras_model": MODELS_DIR / "DSCNN_keyword_spotting.keras",
        "output": MODELS_DIR / "DSCNN_keyword_spotting_int8.tflite",
        "add_channel": True,
    },
]


def quantize_model(item):
    name = item["name"]

    print("\n" + "=" * 60)
    print(f"Quantizing {name}")
    print("=" * 60)

    # Check required files
    config_path = item["config"]
    keras_path = item["keras_model"]

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    # Load training configuration
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Load the original training MFCC features
    loader = MFCCFeatureLoader(
        feature_directory=config["dataset"]["feature_directory"],
        class_names=config["classes"],
    )

    x_train, _ = loader.load_split(
        config["dataset"]["training"]["directory"]
    )

    x_train = np.asarray(x_train, dtype=np.float32)

    if len(x_train) == 0:
        raise ValueError(f"{name}: training feature set is empty.")

    print(f"Raw training feature shape: {x_train.shape}")

    # Reproduce the normalization used by the training pipeline.
    # Do not normalize these features a second time.
    normalizer = FeatureNormalizer()
    x_train = normalizer.fit_transform(x_train)
    x_train = np.asarray(x_train, dtype=np.float32)

    # DS-CNN expects a channel dimension: (71, 13, 1)
    if item["add_channel"]:
        x_train = np.expand_dims(x_train, axis=-1)

    # Load the existing trained model; no training takes place here.
    model = tf.keras.models.load_model(
        keras_path,
        compile=False,
    )

    print(f"Model input shape: {model.input_shape}")
    print(f"Representative feature shape: {x_train.shape[1:]}")

    # Use the representative sample count from the YAML config.
    quantization_config = config.get("quantization", {})
    representative_count = int(
        quantization_config.get("representative_samples", 200)
    )
    representative_count = min(
        representative_count, len(x_train)
    )

    if representative_count < 1:
        raise ValueError("No representative samples available.")

    # Match the original training code's calibration selection.
    representative_features = x_train[:representative_count]

    item["output"].parent.mkdir(parents=True, exist_ok=True)

    # Convert the existing Keras model to fully integer INT8 TFLite.
    TFLiteConverter.convert_int8(
        model,
        representative_features,
        item["output"],
    )

    print(f"Representative samples: {representative_count}")
    print(f"INT8 model saved: {item['output']}")

    # Verify the exported TFLite model's input and output types.
    interpreter = tf.lite.Interpreter(
        model_path=str(item["output"])
    )
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    print(f"TFLite input shape: {input_info['shape']}")
    print(f"TFLite input dtype: {input_info['dtype']}")
    print(f"TFLite output dtype: {output_info['dtype']}")
    print(f"TFLite file size: {item['output'].stat().st_size:,} bytes")

    if (
        input_info["dtype"] != np.int8
        or output_info["dtype"] != np.int8
    ):
        raise RuntimeError(
            f"{name}: input or output is not INT8."
        )

    print(f"{name}: INT8 conversion and verification successful.")


def main():
    for item in MODELS:
        quantize_model(item)

    print("\nAll requested models have been quantized.")


if __name__ == "__main__":
    main()
