import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from src.build_mobile_net_v2_model import (
    build_mobile_net_v2_model,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "mobile_net_v2_training_config.yaml"
)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_feature_dataset(
    feature_directory: Path,
    split: str,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Mel-spectrogram features for one dataset split.
    """

    split_directory = (
        feature_directory / split
    )

    features = []
    labels = []

    for label_index, class_name in enumerate(
        class_names
    ):
        class_directory = (
            split_directory / class_name
        )

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Class directory not found: "
                f"{class_directory}"
            )

        feature_files = sorted(
            class_directory.glob("*.npy")
        )

        for feature_file in feature_files:
            feature = np.load(
                feature_file
            ).astype(np.float32)

            features.append(feature)
            labels.append(label_index)

    if not features:
        raise ValueError(
            f"No feature files found in "
            f"{split_directory}"
        )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def convert_to_mobilenet_input(
    features: np.ndarray,
    target_height: int = 96,
    target_width: int = 96,
) -> np.ndarray:
    """
    Convert log-Mel spectrograms into
    MobileNetV2 input tensors.

    Input:
        (N, frames, mel_bins)

    Output:
        (N, 96, 96, 3)
    """

    if features.ndim != 3:
        raise ValueError(
            "Expected feature shape "
            "(N, frames, mel_bins), "
            f"got {features.shape}"
        )

    features = features[..., np.newaxis]

    features = tf.image.resize(
        features,
        size=(
            target_height,
            target_width,
        ),
        method="bilinear",
    )

    features = tf.repeat(
        features,
        repeats=3,
        axis=-1,
    )

    return features.numpy().astype(
        np.float32
    )


def create_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(features),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def save_json(
    data: dict,
    output_path: Path,
) -> None:
    """Save dictionary as JSON."""

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def main() -> None:

    config = load_config(
        CONFIG_PATH
    )

    seed = config.get(
        "random_seed",
        42,
    )

    set_random_seed(seed)

    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    transfer_config = config[
        "transfer_learning"
    ]
    output_config = config["output"]

    class_names = config["classes"]

    num_classes = model_config[
        "num_classes"
    ]

    input_shape = tuple(
        model_config["input_shape"]
    )

    feature_directory = Path(
        dataset_config[
            "feature_directory"
        ]
    )

    output_directory = Path(
        output_config["directory"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("MobileNetV2 Training")
    print("=" * 60)

    print(
        f"Feature directory : "
        f"{feature_directory}"
    )

    print(
        f"Input shape       : "
        f"{input_shape}"
    )

    print(
        f"Number of classes : "
        f"{num_classes}"
    )

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    print("\nLoading training features...")

    x_train, y_train = load_feature_dataset(
        feature_directory,
        "train",
        class_names,
    )

    print(
        f"Raw training shape: "
        f"{x_train.shape}"
    )

    print("\nLoading validation features...")

    x_val, y_val = load_feature_dataset(
        feature_directory,
        "val",
        class_names,
    )

    print(
        f"Raw validation shape: "
        f"{x_val.shape}"
    )

    print("\nLoading test features...")

    x_test, y_test = load_feature_dataset(
        feature_directory,
        "test",
        class_names,
    )

    print(
        f"Raw test shape: "
        f"{x_test.shape}"
    )

    # ---------------------------------------------------------
    # Convert Mel spectrograms
    # ---------------------------------------------------------

    print(
        "\nConverting features to "
        "MobileNetV2 input..."
    )

    x_train = convert_to_mobilenet_input(
        x_train,
        input_shape[0],
        input_shape[1],
    )

    x_val = convert_to_mobilenet_input(
        x_val,
        input_shape[0],
        input_shape[1],
    )

    x_test = convert_to_mobilenet_input(
        x_test,
        input_shape[0],
        input_shape[1],
    )

    print(
        f"Training input shape: "
        f"{x_train.shape}"
    )

    print(
        f"Validation input shape: "
        f"{x_val.shape}"
    )

    print(
        f"Test input shape: "
        f"{x_test.shape}"
    )

    # ---------------------------------------------------------
    # MobileNetV2 preprocessing
    # ---------------------------------------------------------

    # MobileNetV2 expects input values in
    # approximately [-1, 1].
    #
    # The log-Mel features are normalized globally
    # using the training-set minimum and maximum.

    train_min = np.min(x_train)
    train_max = np.max(x_train)

    if train_max <= train_min:
        raise ValueError(
            "Training features have zero range."
        )

    x_train = (
        2.0
        * (
            x_train - train_min
        )
        / (
            train_max - train_min
        )
        - 1.0
    )

    x_val = (
        2.0
        * (
            x_val - train_min
        )
        / (
            train_max - train_min
        )
        - 1.0
    )

    x_test = (
        2.0
        * (
            x_test - train_min
        )
        / (
            train_max - train_min
        )
        - 1.0
    )

    normalization_path = (
        output_directory
        / output_config[
            "normalization_parameters"
        ]
    )

    np.savez(
        normalization_path,
        minimum=train_min,
        maximum=train_max,
    )

    print(
        "\nNormalization parameters:"
    )

    print(
        f"  Minimum: {train_min:.6f}"
    )

    print(
        f"  Maximum: {train_max:.6f}"
    )

    # ---------------------------------------------------------
    # TensorFlow datasets
    # ---------------------------------------------------------

    batch_size = training_config[
        "batch_size"
    ]

    train_dataset = create_tf_dataset(
        x_train,
        y_train,
        batch_size,
        shuffle=True,
    )

    validation_dataset = create_tf_dataset(
        x_val,
        y_val,
        batch_size,
        shuffle=False,
    )

    test_dataset = create_tf_dataset(
        x_test,
        y_test,
        batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    print("\nBuilding MobileNetV2...")

    model = build_mobile_net_v2_model(
        input_shape=input_shape,
        num_classes=num_classes,
        alpha=transfer_config[
            "alpha"
        ],
        weights=transfer_config[
            "weights"
        ],
        freeze_base=transfer_config[
            "freeze_base"
        ],
    )

    # ---------------------------------------------------------
    # Compile
    # ---------------------------------------------------------

    learning_rate = training_config[
        "learning_rate"
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )

    model.summary()

    # ---------------------------------------------------------
    # Training callbacks
    # ---------------------------------------------------------

    best_model_path = (
        output_directory
        / "MobileNetV2_best.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1.0e-6,
            verbose=1,
        ),
    ]

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_config[
            "epochs"
        ],
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------

    keras_model_path = (
        output_directory
        / output_config[
            "keras_model"
        ]
    )

    model.save(
        keras_model_path
    )

    print(
        f"\nSaved Keras model to:\n"
        f"{keras_model_path}"
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    print("\nEvaluating test set...")

    test_loss, test_accuracy = (
        model.evaluate(
            test_dataset,
            verbose=1,
        )
    )

    print(
        f"\nTest loss     : "
        f"{test_loss:.6f}"
    )

    print(
        f"Test accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------

    history_path = (
        output_directory
        / output_config[
            "training_history"
        ]
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    save_json(
        history_data,
        history_path,
    )

    # ---------------------------------------------------------
    # Save evaluation results
    # ---------------------------------------------------------

    evaluation_path = (
        output_directory
        / output_config[
            "evaluation_results"
        ]
    )

    evaluation_results = {
        "test_loss": float(
            test_loss
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_accuracy_percent": float(
            test_accuracy * 100
        ),
        "test_samples": int(
            len(y_test)
        ),
    }

    save_json(
        evaluation_results,
        evaluation_path,
    )

    # ---------------------------------------------------------
    # Save label mapping
    # ---------------------------------------------------------

    label_mapping_path = (
        output_directory
        / output_config[
            "label_mapping"
        ]
    )

    label_mapping = {
        str(index): class_name
        for index, class_name
        in enumerate(class_names)
    }

    save_json(
        label_mapping,
        label_mapping_path,
    )

    print("\nTraining completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()