import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from src.feature_loader import MFCCFeatureLoader
from src.feature_validator import MFCCFeatureValidator
from src.label_encoder import LabelEncoder
from src.feature_normalizer import FeatureNormalizer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.tf_lite_converter import TFLiteConverter
from src.build_onedcnn_model import build_cnn1d_model


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\configs\oned_cnn_training_config.yaml"
)


# ============================================================
# Utility functions
# ============================================================

def load_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        return yaml.safe_load(file)


def set_random_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)


def print_dataset_summary(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test
):

    print("\n========================================")
    print("Dataset Summary")
    print("========================================")

    print(
        f"Train      : {x_train.shape}"
    )

    print(
        f"Validation : {x_val.shape}"
    )

    print(
        f"Test       : {x_test.shape}"
    )

    print(
        f"Train labels      : {y_train.shape}"
    )

    print(
        f"Validation labels : {y_val.shape}"
    )

    print(
        f"Test labels       : {y_test.shape}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config(
        CONFIG_PATH
    )

    seed = config["random_seed"]

    set_random_seed(seed)

    # --------------------------------------------------------
    # Configuration values
    # --------------------------------------------------------

    class_names = config["classes"]

    input_shape = tuple(
        config["model"]["input_shape"]
    )

    num_classes = config["model"]["num_classes"]

    epochs = config["training"]["epochs"]

    batch_size = config["training"]["batch_size"]

    learning_rate = config["training"]["learning_rate"]

    feature_directory = config[
        "dataset"
    ]["feature_directory"]

    output_directory = Path(
        config["output"]["directory"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check model configuration
    # --------------------------------------------------------

    if input_shape != (71, 13):
        raise ValueError(
            f"Unexpected 1D-CNN input shape: {input_shape}"
        )

    if num_classes != len(class_names):
        raise ValueError(
            "num_classes does not match "
            "the number of class names."
        )

    # --------------------------------------------------------
    # Feature loader
    # --------------------------------------------------------

    loader = MFCCFeatureLoader(
        feature_directory=feature_directory,
        class_names=class_names
    )

    print("\n========================================")
    print("Loading Features")
    print("========================================")

    x_train, y_train_text = loader.load_split(
        config["dataset"]["training"]["directory"]
    )

    x_val, y_val_text = loader.load_split(
        config["dataset"]["validation"]["directory"]
    )

    x_test, y_test_text = loader.load_split(
        config["dataset"]["testing"]["directory"]
    )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    validator = MFCCFeatureValidator(
        expected_shape=(71, 13)
    )

    validator.validate(
        x_train,
        "Training"
    )

    validator.validate(
        x_val,
        "Validation"
    )

    validator.validate(
        x_test,
        "Testing"
    )

    print_dataset_summary(
        x_train,
        y_train_text,
        x_val,
        y_val_text,
        x_test,
        y_test_text
    )

    # --------------------------------------------------------
    # Label encoding
    # --------------------------------------------------------

    encoder = LabelEncoder(
        class_names
    )

    y_train = np.asarray(
        encoder.encode(y_train_text),
        dtype=np.int32
    )

    y_val = np.asarray(
        encoder.encode(y_val_text),
        dtype=np.int32
    )

    y_test = np.asarray(
        encoder.encode(y_test_text),
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    label_mapping_path = (
        output_directory /
        config["output"]["label_mapping"]
    )

    encoder.save(
        label_mapping_path
    )

    # --------------------------------------------------------
    # Normalize features
    # --------------------------------------------------------

    print("\n========================================")
    print("Feature Normalization")
    print("========================================")

    normalizer = FeatureNormalizer()

    x_train = normalizer.fit_transform(
        x_train
    )

    x_val = normalizer.transform(
        x_val
    )

    x_test = normalizer.transform(
        x_test
    )

    normalization_path = (
        output_directory /
        config["output"]["normalization_parameters"]
    )

    normalizer.save(
        normalization_path
    )

    # --------------------------------------------------------
    # No channel expansion for 1D-CNN
    # --------------------------------------------------------

    print("\nAfter feature normalization:")

    print(
        f"x_train: {x_train.shape}"
    )

    print(
        f"x_val:   {x_val.shape}"
    )

    print(
        f"x_test:  {x_test.shape}"
    )

    # --------------------------------------------------------
    # Build 1D-CNN
    # --------------------------------------------------------

    print("\n========================================")
    print("Building 1D-CNN")
    print("========================================")

    model = build_cnn1d_model(
        input_shape=input_shape,
        num_classes=num_classes
    )

    model.summary()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    trainer = ModelTrainer(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    trainer.compile_model(
        model
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n========================================")
    print("Training 1D-CNN")
    print("========================================")

    history = trainer.train(
        model,
        x_train,
        y_train,
        x_val,
        y_val
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        output_directory /
        config["output"]["training_history"]
    )

    trainer.save_history(
        history,
        history_path
    )

    # --------------------------------------------------------
    # Save Keras FP32 model
    # --------------------------------------------------------

    keras_path = (
        output_directory /
        config["output"]["keras_model"]
    )

    model.save(
        keras_path
    )

    print(
        f"\nKeras model saved: {keras_path}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluator = ModelEvaluator()

    results = evaluator.evaluate(
        model,
        x_test,
        y_test,
        class_names
    )

    evaluator.print_results(
        results
    )

    evaluation_path = (
        output_directory /
        config["output"]["evaluation_results"]
    )

    evaluator.save_results(
        results,
        evaluation_path
    )

    # --------------------------------------------------------
    # FP32 TFLite
    # --------------------------------------------------------

    fp32_path = (
        output_directory /
        config["output"]["fp32_tflite"]
    )

    TFLiteConverter.convert_fp32(
        model,
        fp32_path
    )

    # --------------------------------------------------------
    # INT8 representative dataset
    # --------------------------------------------------------

    representative_count = config[
        "quantization"
    ]["representative_samples"]

    representative_count = min(
        representative_count,
        len(x_train)
    )

    representative_features = (
        x_train[:representative_count]
    )

    # --------------------------------------------------------
    # INT8 TFLite
    # --------------------------------------------------------

    int8_path = (
        output_directory /
        config["output"]["int8_tflite"]
    )

    TFLiteConverter.convert_int8(
        model,
        representative_features,
        int8_path
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n========================================")
    print("Training Completed")
    print("========================================")

    print(
        f"Parameters: "
        f"{model.count_params():,}"
    )

    print(
        f"Keras model : {keras_path}"
    )

    print(
        f"FP32 TFLite : {fp32_path}"
    )

    print(
        f"INT8 TFLite : {int8_path}"
    )


if __name__ == "__main__":
    main()