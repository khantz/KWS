
"""
evaluate_tflite_models.py

Evaluate CNN1D, DSCNN, and MobileNetV2 TFLite models
in FP32 and INT8 formats.

Project structure:

KWS/
├── Models/
│   ├── CNN1D_keyword_spotting_fp32.tflite
│   ├── CNN1D_keyword_spotting_int8.tflite
│   ├── CNN1D_normalization.npz
│   ├── CNN1D_label_mapping.json
│   │
│   ├── DSCNN_keyword_spotting_fp32.tflite
│   ├── DSCNN_keyword_spotting_int8.tflite
│   ├── DSCNN_normalization.npz
│   ├── DSCNN_label_mapping.json
│   │
│   ├── MobileNetV2_keyword_spotting.keras
│   ├── MobileNetV2_keyword_spotting_fp32.tflite
│   ├── MobileNetV2_keyword_spotting_int8.tflite
│   ├── MobileNetV2_normalization.npz
│   └── MobileNetV2_label_mapping.json
│
├── feature/
│   ├── mfcc_feature/
│   │   └── test/
│   │       ├── no/
│   │       ├── off/
│   │       └── ...
│   │
│   └── mel_spectrogram/
│       └── test/
│           ├── no/
│           ├── off/
│           └── ...
│
└── src/
    └── scripts/
        └── evaluate_tflite_models.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\M S I\PycharmProjects\KWS")

MODELS_DIR = PROJECT_ROOT / "Models"

MFCC_TEST_DIR = (
    PROJECT_ROOT
    / "feature"
    / "mfcc_feature"
    / "test"
)

MEL_TEST_DIR = (
    PROJECT_ROOT
    / "feature"
    / "mel_spectrogram"
    / "test"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation_results"
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    "CNN1D": {
        "fp32": MODELS_DIR / "CNN1D_keyword_spotting_fp32.tflite",
        "int8": MODELS_DIR / "CNN1D_keyword_spotting_int8.tflite",
        "normalization": MODELS_DIR / "CNN1D_normalization.npz",
        "label_mapping": MODELS_DIR / "CNN1D_label_mapping.json",
        "feature_type": "mfcc",
    },

    "DSCNN": {
        "fp32": MODELS_DIR / "DSCNN_keyword_spotting_fp32.tflite",
        "int8": MODELS_DIR / "DSCNN_keyword_spotting_int8.tflite",
        "normalization": MODELS_DIR / "DSCNN_normalization.npz",
        "label_mapping": MODELS_DIR / "DSCNN_label_mapping.json",
        "feature_type": "mfcc",
    },

    "MobileNetV2": {
        "fp32": MODELS_DIR / "mobilenetv2_fp32.tflite",
        "int8": MODELS_DIR / "mobilenetv2_int8.tflite",
        "normalization": MODELS_DIR / "MobileNetV2_normalization.npz",
        "label_mapping": MODELS_DIR / "MobileNetV2_label_mapping.json",
        "feature_type": "mel",
    },
}


# ============================================================================
# GENERAL SETTINGS
# ============================================================================

PRINT_EVERY = 500

MOBILENET_IMAGE_SIZE = (96, 96)

EXPECTED_MFCC_SHAPE = (71, 13)
EXPECTED_MEL_SHAPE = (71, 40)


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def json_serializer(obj):
    """
    Convert NumPy/path objects into JSON-compatible objects.
    """

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, Path):
        return str(obj)

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )


def save_json(data, path):
    """
    Save Python/NumPy data as JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=json_serializer,
        )


# ============================================================================
# LABEL MAPPING
# ============================================================================

def load_label_mapping(path):
    """
    Load model label mapping.

    Supports common formats such as:

        {
            "0": "no",
            "1": "off",
            ...
        }

    or:

        {
            "no": 0,
            "off": 1,
            ...
        }

    or a list:

        [
            "no",
            "off",
            ...
        ]
    """

    path = Path(path)

    print()
    print("Loading label mapping:")
    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nLabel mapping file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):

        index_to_label = {
            int(index): str(label)
            for index, label in enumerate(data)
        }

    elif isinstance(data, dict):

        # ------------------------------------------------------------
        # Case 1:
        # {"0": "no", "1": "off", ...}
        # ------------------------------------------------------------

        if all(
            str(key).isdigit()
            for key in data.keys()
        ):
            index_to_label = {
                int(key): str(value)
                for key, value in data.items()
            }

        # ------------------------------------------------------------
        # Case 2:
        # {"no": 0, "off": 1, ...}
        # ------------------------------------------------------------

        elif all(
            isinstance(value, (int, np.integer))
            for value in data.values()
        ):
            index_to_label = {
                int(value): str(key)
                for key, value in data.items()
            }

        # ------------------------------------------------------------
        # Search for nested mappings
        # ------------------------------------------------------------

        else:

            possible_keys = [
                "index_to_label",
                "label_mapping",
                "class_names",
                "classes",
            ]

            nested = None

            for key in possible_keys:
                if key in data:
                    nested = data[key]
                    break

            if nested is None:
                raise ValueError(
                    f"\nUnable to understand label mapping:\n{path}\n"
                    f"Contents: {data}"
                )

            if isinstance(nested, list):
                index_to_label = {
                    int(index): str(label)
                    for index, label in enumerate(nested)
                }

            elif isinstance(nested, dict):

                if all(
                    str(key).isdigit()
                    for key in nested.keys()
                ):
                    index_to_label = {
                        int(key): str(value)
                        for key, value in nested.items()
                    }

                else:
                    index_to_label = {
                        int(value): str(key)
                        for key, value in nested.items()
                    }

            else:
                raise ValueError(
                    f"\nUnsupported label mapping format:\n{path}"
                )

    else:
        raise ValueError(
            f"\nUnsupported label mapping format:\n{path}"
        )

    index_to_label = dict(
        sorted(index_to_label.items())
    )

    print(
        f"  Number of classes: "
        f"{len(index_to_label)}"
    )

    for index, label in index_to_label.items():
        print(
            f"  {index}: {label}"
        )

    return index_to_label


# ============================================================================
# NORMALIZATION
# ============================================================================

def load_normalization(path):
    """
    Load normalization parameters.

    Supported formats:

    1. Mean/std:

        mean
        std

        normalized = (x - mean) / std

    2. Minimum/maximum:

        minimum
        maximum

        normalized =
            2 * (x - minimum) / (maximum - minimum) - 1
    """

    path = Path(path)

    print()
    print("Loading normalization:")
    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nNormalization file not found:\n{path}"
        )

    data = np.load(path)

    keys = list(data.keys())

    print(
        f"  Keys: {keys}"
    )

    # ----------------------------------------------------------------
    # Mean / Std
    # ----------------------------------------------------------------

    if (
        "mean" in data
        and "std" in data
    ):

        mean = np.asarray(
            data["mean"],
            dtype=np.float32,
        )

        std = np.asarray(
            data["std"],
            dtype=np.float32,
        )

        if np.any(std == 0):
            raise ValueError(
                f"\nNormalization std contains zero:\n{path}"
            )

        print(
            "  Type: mean/std standardization"
        )

        print(
            f"  mean shape: {mean.shape}"
        )

        print(
            f"  std shape : {std.shape}"
        )

        return {
            "type": "mean_std",
            "mean": mean,
            "std": std,
        }

    # ----------------------------------------------------------------
    # Minimum / Maximum
    # ----------------------------------------------------------------

    if (
        "minimum" in data
        and "maximum" in data
    ):

        minimum = np.asarray(
            data["minimum"],
            dtype=np.float32,
        )

        maximum = np.asarray(
            data["maximum"],
            dtype=np.float32,
        )

        if np.any(maximum == minimum):
            raise ValueError(
                f"\nNormalization maximum equals minimum:\n{path}"
            )

        print(
            "  Type: min/max normalization"
        )

        print(
            f"  minimum shape: {minimum.shape}"
        )

        print(
            f"  maximum shape: {maximum.shape}"
        )

        return {
            "type": "min_max",
            "minimum": minimum,
            "maximum": maximum,
        }

    raise ValueError(
        "\nUnknown normalization format.\n"
        f"File: {path}\n"
        f"Available keys: {keys}\n\n"
        "Expected either:\n"
        "  mean + std\n"
        "or:\n"
        "  minimum + maximum"
    )


def normalize_features(
    features,
    normalization,
):
    """
    Apply the saved training normalization.

    IMPORTANT:

    Normalization is applied while the feature is still:

        (N, 71, 13)

    For DSCNN, the channel dimension is added
    AFTER this function.

    This prevents:

        (N, 71, 13, 1)
        - (1, 13)

    from incorrectly broadcasting to:

        (N, 71, 13, 13)
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    normalization_type = normalization["type"]

    # ----------------------------------------------------------------
    # Mean / Std
    # ----------------------------------------------------------------

    if normalization_type == "mean_std":

        mean = normalization["mean"]
        std = normalization["std"]

        normalized = (
            features - mean
        ) / std

        return normalized.astype(
            np.float32
        )

    # ----------------------------------------------------------------
    # Min / Max
    # ----------------------------------------------------------------

    if normalization_type == "min_max":

        minimum = normalization["minimum"]
        maximum = normalization["maximum"]

        normalized = (
            2.0
            * (features - minimum)
            / (maximum - minimum)
            - 1.0
        )

        return normalized.astype(
            np.float32
        )

    raise ValueError(
        f"Unknown normalization type: "
        f"{normalization_type}"
    )


# ============================================================================
# FEATURE LOADING
# ============================================================================

def collect_npy_files(directory):
    """
    Collect .npy files from class subdirectories.

    Returns:

        [
            (file_path, class_name),
            ...
        ]
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"\nFeature directory not found:\n{directory}"
        )

    files = []

    class_dirs = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name.lower(),
    )

    if not class_dirs:
        raise ValueError(
            f"\nNo class directories found:\n{directory}"
        )

    for class_dir in class_dirs:

        class_name = class_dir.name

        npy_files = sorted(
            class_dir.glob("*.npy")
        )

        for file_path in npy_files:
            files.append(
                (
                    file_path,
                    class_name,
                )
            )

    if not files:
        raise ValueError(
            f"\nNo .npy files found:\n{directory}"
        )

    return files


def load_feature_dataset(
    directory,
    feature_name,
):
    """
    Load an entire feature test dataset.

    Labels are obtained from the parent directory name.

    Returns:

        features
        labels
        paths
    """

    directory = Path(directory)

    print()
    print(
        f"Loading {feature_name} test features..."
    )

    print(
        f"Directory: {directory}"
    )

    files = collect_npy_files(
        directory
    )

    features = []
    labels = []
    paths = []

    for file_path, class_name in files:

        feature = np.load(
            file_path
        )

        feature = np.asarray(
            feature,
            dtype=np.float32,
        )

        features.append(feature)
        labels.append(class_name)
        paths.append(str(file_path))

    try:

        features = np.stack(
            features,
            axis=0,
        )

    except ValueError as exc:

        raise ValueError(
            f"\nUnable to stack {feature_name} "
            "features.\n"
            "All feature files must have "
            "the same shape."
        ) from exc

    labels = np.asarray(
        labels,
        dtype=str,
    )

    print(
        f"{feature_name} shape: "
        f"{features.shape}"
    )

    print(
        f"Number of samples: "
        f"{len(labels)}"
    )

    # ----------------------------------------------------------------
    # Class distribution
    # ----------------------------------------------------------------

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    print(
        f"{feature_name} class distribution:"
    )

    for label, count in zip(
        unique_labels,
        counts,
    ):
        print(
            f"  {label:<12} {count}"
        )

    return (
        features,
        labels,
        paths,
    )


# ============================================================================
# LABEL VERIFICATION
# ============================================================================

def verify_feature_labels(
    mfcc_labels,
    mel_labels,
    mfcc_paths,
    mel_paths,
):
    """
    Verify MFCC and Mel datasets contain
    the same samples/classes in the same order.
    """

    if len(mfcc_labels) != len(mel_labels):

        raise ValueError(
            "\nMFCC and Mel test datasets "
            "have different sample counts.\n"
            f"MFCC: {len(mfcc_labels)}\n"
            f"Mel : {len(mel_labels)}"
        )

    if not np.array_equal(
        mfcc_labels,
        mel_labels,
    ):

        print()
        print(
            "WARNING: MFCC and Mel label arrays "
            "are not identical."
        )

        mismatch_indices = np.where(
            mfcc_labels != mel_labels
        )[0]

        print(
            f"Number of mismatches: "
            f"{len(mismatch_indices)}"
        )

        for index in mismatch_indices[:10]:

            print(
                f"  Sample {index}:"
            )

            print(
                f"    MFCC: "
                f"{mfcc_labels[index]}"
            )

            print(
                f"    Mel : "
                f"{mel_labels[index]}"
            )

            print(
                f"    MFCC file: "
                f"{mfcc_paths[index]}"
            )

            print(
                f"    Mel file : "
                f"{mel_paths[index]}"
            )

        raise ValueError(
            "\nMFCC and Mel test labels do not agree."
        )

    print()
    print(
        "MFCC and Mel test labels agree."
    )


# ============================================================================
# MODEL INPUT PREPARATION
# ============================================================================

def prepare_cnn1d_features(
    mfcc_features,
    normalization,
):
    """
    CNN1D expected input:

        (N, 71, 13)

    Steps:

        MFCC
        (N, 71, 13)
            ↓
        normalization
        (N, 71, 13)
    """

    features = np.asarray(
        mfcc_features,
        dtype=np.float32,
    )

    if features.ndim != 3:
        raise ValueError(
            "\nCNN1D MFCC feature must be 3D.\n"
            f"Actual shape: {features.shape}"
        )

    if tuple(features.shape[1:]) != EXPECTED_MFCC_SHAPE:

        raise ValueError(
            "\nUnexpected CNN1D MFCC shape.\n"
            f"Expected: {EXPECTED_MFCC_SHAPE}\n"
            f"Actual: {features.shape[1:]}"
        )

    # IMPORTANT:
    # Normalize BEFORE adding any channel dimension.
    features = normalize_features(
        features,
        normalization,
    )

    return features


def prepare_dscnn_features(
    mfcc_features,
    normalization,
):
    """
    DSCNN expected input:

        (N, 71, 13, 1)

    IMPORTANT:

        Normalize first:

            (N, 71, 13)
                ↓
            (N, 71, 13)

        Then add channel:

            (N, 71, 13)
                ↓
            (N, 71, 13, 1)

    This prevents the previous broadcasting bug:

        (N, 71, 13, 1)
        - (1, 13)
        =
        (N, 71, 13, 13)   <-- WRONG
    """

    features = np.asarray(
        mfcc_features,
        dtype=np.float32,
    )

    if features.ndim != 3:
        raise ValueError(
            "\nDSCNN MFCC feature must initially be 3D.\n"
            f"Actual shape: {features.shape}"
        )

    if tuple(features.shape[1:]) != EXPECTED_MFCC_SHAPE:

        raise ValueError(
            "\nUnexpected DSCNN MFCC shape.\n"
            f"Expected: {EXPECTED_MFCC_SHAPE}\n"
            f"Actual: {features.shape[1:]}"
        )

    # ------------------------------------------------------------
    # STEP 1: Normalize while shape is (N, 71, 13)
    # ------------------------------------------------------------

    features = normalize_features(
        features,
        normalization,
    )

    # ------------------------------------------------------------
    # STEP 2: Add channel dimension
    #
    # (N, 71, 13)
    #       ↓
    # (N, 71, 13, 1)
    # ------------------------------------------------------------

    features = features[
        ..., np.newaxis
    ]

    if features.ndim != 4:
        raise RuntimeError(
            "\nDSCNN feature preparation failed.\n"
            f"Expected 4 dimensions.\n"
            f"Actual: {features.shape}"
        )

    if tuple(features.shape[1:]) != (
        71,
        13,
        1,
    ):

        raise RuntimeError(
            "\nDSCNN feature shape is incorrect.\n"
            "Expected:\n"
            "  (N, 71, 13, 1)\n"
            f"Actual:\n"
            f"  {features.shape}"
        )

    return features


def resize_mel_features(
    mel_features,
):
    """
    Convert:

        (N, 71, 40)

    to:

        (N, 96, 96)

    using TensorFlow bilinear resizing.
    """

    mel_features = np.asarray(
        mel_features,
        dtype=np.float32,
    )

    if mel_features.ndim != 3:
        raise ValueError(
            "\nMel feature must be 3D.\n"
            f"Actual shape: {mel_features.shape}"
        )

    if tuple(mel_features.shape[1:]) != EXPECTED_MEL_SHAPE:

        raise ValueError(
            "\nUnexpected Mel feature shape.\n"
            f"Expected: {EXPECTED_MEL_SHAPE}\n"
            f"Actual: {mel_features.shape[1:]}"
        )

    # Add single channel.
    #
    # (N, 71, 40)
    #       ↓
    # (N, 71, 40, 1)

    mel_features = mel_features[
        ..., np.newaxis
    ]

    # Resize.
    #
    # (N, 71, 40, 1)
    #       ↓
    # (N, 96, 96, 1)

    resized = tf.image.resize(
        mel_features,
        MOBILENET_IMAGE_SIZE,
        method="bilinear",
    )

    resized = resized.numpy()

    return resized.astype(
        np.float32
    )


def prepare_mobilenet_features(
    mel_features,
    normalization,
):
    """
    MobileNetV2 expected input:

        (N, 96, 96, 3)

    Training pipeline:

        Mel
        (N, 71, 40)
            ↓
        resize
        (N, 96, 96, 1)
            ↓
        repeat channel
        (N, 96, 96, 3)
            ↓
        min/max normalization
        [-1, 1]
    """

    features = np.asarray(
        mel_features,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    features = resize_mel_features(
        features
    )

    # ------------------------------------------------------------
    # Convert grayscale -> RGB
    #
    # (N, 96, 96, 1)
    #       ↓
    # (N, 96, 96, 3)
    # ------------------------------------------------------------

    features = np.repeat(
        features,
        repeats=3,
        axis=-1,
    )

    # ------------------------------------------------------------
    # Apply saved normalization
    # ------------------------------------------------------------

    features = normalize_features(
        features,
        normalization,
    )

    return features


# ============================================================================
# TFLITE HELPERS
# ============================================================================

def get_tensor_details(
    interpreter,
):
    """
    Get TFLite input/output details.
    """

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    if len(input_details) != 1:
        raise ValueError(
            "Expected exactly one TFLite input tensor."
        )

    if len(output_details) != 1:
        raise ValueError(
            "Expected exactly one TFLite output tensor."
        )

    return (
        input_details[0],
        output_details[0],
    )


def get_quantization_parameters(
    tensor_details,
):
    """
    Return:

        scale
        zero_point
    """

    quantization = tensor_details.get(
        "quantization",
        (0.0, 0),
    )

    scale = float(
        quantization[0]
    )

    zero_point = int(
        quantization[1]
    )

    return (
        scale,
        zero_point,
    )


def quantize_input(
    features,
    input_details,
):
    """
    Convert normalized float input to the
    datatype required by the TFLite model.
    """

    dtype = input_details["dtype"]

    # ------------------------------------------------------------
    # Float model
    # ------------------------------------------------------------

    if dtype in (
        np.float32,
        np.float64,
    ):

        return features.astype(
            dtype
        )

    # ------------------------------------------------------------
    # Quantized model
    # ------------------------------------------------------------

    if dtype in (
        np.int8,
        np.uint8,
    ):

        scale, zero_point = (
            get_quantization_parameters(
                input_details
            )
        )

        if scale == 0:
            raise ValueError(
                "\nTFLite input quantization "
                "scale is zero."
            )

        quantized = (
            np.round(
                features / scale
            )
            + zero_point
        )

        if dtype == np.int8:

            quantized = np.clip(
                quantized,
                -128,
                127,
            )

        elif dtype == np.uint8:

            quantized = np.clip(
                quantized,
                0,
                255,
            )

        return quantized.astype(
            dtype
        )

    raise ValueError(
        "\nUnsupported TFLite input dtype: "
        f"{dtype}"
    )


def dequantize_output(
    output,
    output_details,
):
    """
    Convert TFLite output to float32.

    For float models, return output directly.

    For int8/uint8 models:

        real_value =
            (integer_value - zero_point)
            * scale
    """

    dtype = output_details["dtype"]

    if dtype in (
        np.float32,
        np.float64,
    ):

        return np.asarray(
            output,
            dtype=np.float32,
        )

    if dtype in (
        np.int8,
        np.uint8,
    ):

        scale, zero_point = (
            get_quantization_parameters(
                output_details
            )
        )

        if scale == 0:
            raise ValueError(
                "\nTFLite output quantization "
                "scale is zero."
            )

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        return (
            output - zero_point
        ) * scale

    raise ValueError(
        "\nUnsupported TFLite output dtype: "
        f"{dtype}"
    )


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(
    interpreter,
    features,
):
    """
    Run TFLite inference sample by sample.

    The input feature array does not contain
    the batch dimension expected by the model.

    Example:

        features:
            (N, 71, 13)

        model input:
            (1, 71, 13)
    """

    input_details, output_details = (
        get_tensor_details(
            interpreter
        )
    )

    expected_shape = tuple(
        int(x)
        for x in input_details["shape"]
    )

    if len(expected_shape) < 2:
        raise ValueError(
            "\nUnexpected TFLite input shape:\n"
            f"{expected_shape}"
        )

    expected_sample_shape = (
        expected_shape[1:]
    )

    actual_sample_shape = tuple(
        features.shape[1:]
    )

    print(
        f"Expected input shape: "
        f"{expected_shape}"
    )

    print(
        f"Prepared feature shape: "
        f"{features.shape}"
    )

    if (
        actual_sample_shape
        != expected_sample_shape
    ):

        raise ValueError(
            "\nInput shape mismatch.\n"
            f"Expected: {expected_sample_shape}\n"
            f"Actual: {actual_sample_shape}"
        )

    number_of_samples = (
        features.shape[0]
    )

    predictions = []

    print(
        f"Running inference on "
        f"{number_of_samples} samples..."
    )

    start_time = time.perf_counter()

    for index in range(
        number_of_samples
    ):

        sample = features[
            index:index + 1
        ]

        # --------------------------------------------------------
        # Convert float features to model input dtype.
        # --------------------------------------------------------

        input_data = quantize_input(
            sample,
            input_details,
        )

        interpreter.set_tensor(
            input_details["index"],
            input_data,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        output = dequantize_output(
            output,
            output_details,
        )

        prediction = int(
            np.argmax(
                output[0]
            )
        )

        predictions.append(
            prediction
        )

        completed = index + 1

        if (
            completed % PRINT_EVERY == 0
            or completed == number_of_samples
        ):

            print(
                f"  {completed}/"
                f"{number_of_samples}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    return (
        predictions,
        elapsed,
    )


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    y_pred,
    index_to_label,
):
    """
    Calculate accuracy, macro F1,
    weighted F1, classification report,
    and confusion matrix.
    """

    class_indices = sorted(
        index_to_label.keys()
    )

    class_names = [
        index_to_label[index]
        for index in class_indices
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_indices,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_indices,
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "classification_report": report,

        "confusion_matrix": matrix,

        "class_indices": class_indices,

        "class_names": class_names,
    }


# ============================================================================
# MODEL SIZE
# ============================================================================

def get_file_size_kb(
    path,
):
    """
    Return model file size in KB.
    """

    path = Path(path)

    if not path.exists():
        return None

    return (
        path.stat().st_size
        / 1024.0
    )


# ============================================================================
# SINGLE MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model_name,
    quantization_type,
    model_path,
    normalization_path,
    features,
    true_label_names,
    index_to_label,
):
    """
    Evaluate one TFLite model.
    """

    print()
    print(
        "=" * 70
    )

    print(
        f"{model_name} - "
        f"{quantization_type.upper()}"
    )

    print(
        "=" * 70
    )

    print(
        f"Model path       : "
        f"{model_path}"
    )

    # ------------------------------------------------------------
    # Load normalization
    # ------------------------------------------------------------

    normalization = (
        load_normalization(
            normalization_path
        )
    )

    # ------------------------------------------------------------
    # Prepare model-specific features
    # ------------------------------------------------------------

    if model_name == "CNN1D":

        prepared_features = (
            prepare_cnn1d_features(
                features,
                normalization,
            )
        )

    elif model_name == "DSCNN":

        prepared_features = (
            prepare_dscnn_features(
                features,
                normalization,
            )
        )

    elif model_name == "MobileNetV2":

        prepared_features = (
            prepare_mobilenet_features(
                features,
                normalization,
            )
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    print()
    print(
        f"Final feature shape: "
        f"{prepared_features.shape}"
    )

    print(
        f"Feature range: "
        f"{prepared_features.min():.6f} "
        f"to "
        f"{prepared_features.max():.6f}"
    )

    # ------------------------------------------------------------
    # Load TFLite model
    # ------------------------------------------------------------

    if not Path(model_path).exists():

        raise FileNotFoundError(
            f"\nTFLite model not found:\n"
            f"{model_path}"
        )

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details, output_details = (
        get_tensor_details(
            interpreter
        )
    )

    input_shape = tuple(
        int(x)
        for x in input_details["shape"]
    )

    output_shape = tuple(
        int(x)
        for x in output_details["shape"]
    )

    input_scale, input_zero = (
        get_quantization_parameters(
            input_details
        )
    )

    output_scale, output_zero = (
        get_quantization_parameters(
            output_details
        )
    )

    print(
        f"Input shape      : "
        f"{input_shape}"
    )

    print(
        f"Input dtype      : "
        f"{input_details['dtype']}"
    )

    print(
        f"Input scale      : "
        f"{input_scale}"
    )

    print(
        f"Input zero point : "
        f"{input_zero}"
    )

    print(
        f"Output shape     : "
        f"{output_shape}"
    )

    print(
        f"Output dtype     : "
        f"{output_details['dtype']}"
    )

    print(
        f"Output scale     : "
        f"{output_scale}"
    )

    print(
        f"Output zero point: "
        f"{output_zero}"
    )

    # ------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------

    predictions, inference_time = (
        run_inference(
            interpreter,
            prepared_features,
        )
    )

    # ------------------------------------------------------------
    # Convert class names to indices
    # ------------------------------------------------------------

    label_to_index = {
        label: index
        for index, label
        in index_to_label.items()
    }

    try:

        y_true = np.asarray(
            [
                label_to_index[
                    label
                ]
                for label
                in true_label_names
            ],
            dtype=np.int64,
        )

    except KeyError as exc:

        raise ValueError(
            "\nTest dataset contains a label "
            "that is not present in the model "
            "label mapping.\n"
            f"Unknown label: {exc}"
        ) from exc

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        predictions,
        index_to_label,
    )

    accuracy = metrics[
        "accuracy"
    ]

    macro_f1 = metrics[
        "macro_f1"
    ]

    weighted_f1 = metrics[
        "weighted_f1"
    ]

    model_size_kb = (
        get_file_size_kb(
            model_path
        )
    )

    average_latency_ms = (
        inference_time
        / len(prepared_features)
        * 1000.0
    )

    print()
    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Model size: "
        f"{model_size_kb:.2f} KB"
    )

    print(
        f"Total inference time: "
        f"{inference_time:.4f} s"
    )

    print(
        f"Average inference time: "
        f"{average_latency_ms:.4f} ms/sample"
    )

    # ------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------

    result = {
        "model": model_name,

        "quantization": quantization_type,

        "model_path": str(
            model_path
        ),

        "normalization_path": str(
            normalization_path
        ),

        "normalization_type": (
            normalization["type"]
        ),

        "input_shape": input_shape,

        "input_dtype": str(
            input_details["dtype"]
        ),

        "input_scale": input_scale,

        "input_zero_point": input_zero,

        "output_shape": output_shape,

        "output_dtype": str(
            output_details["dtype"]
        ),

        "output_scale": output_scale,

        "output_zero_point": output_zero,

        "feature_shape": tuple(
            prepared_features.shape
        ),

        "feature_min": float(
            prepared_features.min()
        ),

        "feature_max": float(
            prepared_features.max()
        ),

        "test_samples": int(
            len(y_true)
        ),

        "accuracy": accuracy,

        "macro_f1": macro_f1,

        "weighted_f1": weighted_f1,

        "model_size_kb": model_size_kb,

        "total_inference_time_seconds": (
            inference_time
        ),

        "average_inference_time_ms": (
            average_latency_ms
        ),

        "classification_report": (
            metrics[
                "classification_report"
            ]
        ),

        "confusion_matrix": (
            metrics[
                "confusion_matrix"
            ]
        ),

        "class_indices": (
            metrics[
                "class_indices"
            ]
        ),

        "class_names": (
            metrics[
                "class_names"
            ]
        ),
    }

    return result


# ============================================================================
# LABEL MAPPING VERIFICATION
# ============================================================================

def verify_label_mappings(
    mappings,
):
    """
    Verify that all models use the same
    class-index mapping.
    """

    names = list(
        mappings.keys()
    )

    if not names:
        raise ValueError(
            "No label mappings loaded."
        )

    reference_name = names[0]

    reference = mappings[
        reference_name
    ]

    for name in names[1:]:

        current = mappings[name]

        if current != reference:

            raise ValueError(
                "\nLabel mapping mismatch.\n"
                f"{reference_name}: "
                f"{reference}\n"
                f"{name}: "
                f"{current}"
            )

    print()
    print(
        "All label mappings agree."
    )


# ============================================================================
# SUMMARY
# ============================================================================

def print_final_summary(
    results,
):
    """
    Print a concise final comparison.
    """

    print()
    print()
    print(
        "=" * 90
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 90
    )

    header = (
        f"{'Model':<16}"
        f"{'Type':<10}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>14}"
        f"{'Size KB':>12}"
    )

    print(header)

    print(
        "-" * 90
    )

    for result in results:

        print(
            f"{result['model']:<16}"
            f"{result['quantization']:<10}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['macro_f1']:>12.4f}"
            f"{result['weighted_f1']:>14.4f}"
            f"{result['model_size_kb']:>12.2f}"
        )

    print(
        "=" * 90
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TFLite FP32 / INT8 Evaluation"
    )

    print(
        "=" * 70
    )

    # ----------------------------------------------------------------
    # Check directories
    # ----------------------------------------------------------------

    if not PROJECT_ROOT.exists():

        raise FileNotFoundError(
            f"\nProject root not found:\n"
            f"{PROJECT_ROOT}"
        )

    if not MODELS_DIR.exists():

        raise FileNotFoundError(
            f"\nModels directory not found:\n"
            f"{MODELS_DIR}"
        )

    # ----------------------------------------------------------------
    # Load all label mappings
    # ----------------------------------------------------------------

    label_mappings = {}

    for model_name, config in (
        MODEL_CONFIGS.items()
    ):

        label_mappings[
            model_name
        ] = load_label_mapping(
            config["label_mapping"]
        )

    verify_label_mappings(
        label_mappings
    )

    # All models use the same mapping.
    index_to_label = (
        label_mappings[
            "CNN1D"
        ]
    )

    # ----------------------------------------------------------------
    # Load MFCC test data
    # ----------------------------------------------------------------

    (
        mfcc_features,
        mfcc_labels,
        mfcc_paths,
    ) = load_feature_dataset(
        MFCC_TEST_DIR,
        "MFCC",
    )

    # ----------------------------------------------------------------
    # Load Mel test data
    # ----------------------------------------------------------------

    (
        mel_features,
        mel_labels,
        mel_paths,
    ) = load_feature_dataset(
        MEL_TEST_DIR,
        "Mel",
    )

    # ----------------------------------------------------------------
    # Verify MFCC/Mel labels
    # ----------------------------------------------------------------

    verify_feature_labels(
        mfcc_labels,
        mel_labels,
        mfcc_paths,
        mel_paths,
    )

    # ----------------------------------------------------------------
    # Check feature counts
    # ----------------------------------------------------------------

    if (
        mfcc_features.shape[0]
        != mel_features.shape[0]
    ):

        raise ValueError(
            "\nMFCC and Mel sample counts differ.\n"
            f"MFCC: {mfcc_features.shape[0]}\n"
            f"Mel : {mel_features.shape[0]}"
        )

    # ----------------------------------------------------------------
    # Evaluate all models
    # ----------------------------------------------------------------

    results = []

    # ================================================================
    # CNN1D
    # ================================================================

    cnn1d_fp32_result = evaluate_model(
        model_name="CNN1D",
        quantization_type="FP32",
        model_path=MODEL_CONFIGS[
            "CNN1D"
        ]["fp32"],
        normalization_path=MODEL_CONFIGS[
            "CNN1D"
        ]["normalization"],
        features=mfcc_features,
        true_label_names=mfcc_labels,
        index_to_label=index_to_label,
    )

    results.append(
        cnn1d_fp32_result
    )

    cnn1d_int8_result = evaluate_model(
        model_name="CNN1D",
        quantization_type="INT8",
        model_path=MODEL_CONFIGS[
            "CNN1D"
        ]["int8"],
        normalization_path=MODEL_CONFIGS[
            "CNN1D"
        ]["normalization"],
        features=mfcc_features,
        true_label_names=mfcc_labels,
        index_to_label=index_to_label,
    )

    results.append(
        cnn1d_int8_result
    )

    # ================================================================
    # DSCNN
    # ================================================================

    dscnn_fp32_result = evaluate_model(
        model_name="DSCNN",
        quantization_type="FP32",
        model_path=MODEL_CONFIGS[
            "DSCNN"
        ]["fp32"],
        normalization_path=MODEL_CONFIGS[
            "DSCNN"
        ]["normalization"],
        features=mfcc_features,
        true_label_names=mfcc_labels,
        index_to_label=index_to_label,
    )

    results.append(
        dscnn_fp32_result
    )

    dscnn_int8_result = evaluate_model(
        model_name="DSCNN",
        quantization_type="INT8",
        model_path=MODEL_CONFIGS[
            "DSCNN"
        ]["int8"],
        normalization_path=MODEL_CONFIGS[
            "DSCNN"
        ]["normalization"],
        features=mfcc_features,
        true_label_names=mfcc_labels,
        index_to_label=index_to_label,
    )

    results.append(
        dscnn_int8_result
    )

    # ================================================================
    # MobileNetV2
    # ================================================================

    mobilenet_fp32_result = evaluate_model(
        model_name="MobileNetV2",
        quantization_type="FP32",
        model_path=MODEL_CONFIGS[
            "MobileNetV2"
        ]["fp32"],
        normalization_path=MODEL_CONFIGS[
            "MobileNetV2"
        ]["normalization"],
        features=mel_features,
        true_label_names=mel_labels,
        index_to_label=index_to_label,
    )

    results.append(
        mobilenet_fp32_result
    )

    mobilenet_int8_result = evaluate_model(
        model_name="MobileNetV2",
        quantization_type="INT8",
        model_path=MODEL_CONFIGS[
            "MobileNetV2"
        ]["int8"],
        normalization_path=MODEL_CONFIGS[
            "MobileNetV2"
        ]["normalization"],
        features=mel_features,
        true_label_names=mel_labels,
        index_to_label=index_to_label,
    )

    results.append(
        mobilenet_int8_result
    )

    # ----------------------------------------------------------------
    # Print final summary
    # ----------------------------------------------------------------

    print_final_summary(
        results
    )

    # ----------------------------------------------------------------
    # Save complete results
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "tflite_evaluation_results.json"
    )

    complete_results = {
        "project_root": str(
            PROJECT_ROOT
        ),

        "models_directory": str(
            MODELS_DIR
        ),

        "mfcc_test_directory": str(
            MFCC_TEST_DIR
        ),

        "mel_test_directory": str(
            MEL_TEST_DIR
        ),

        "test_samples": int(
            len(mfcc_labels)
        ),

        "class_mapping": index_to_label,

        "results": results,
    }

    save_json(
        complete_results,
        output_path,
    )

    print()
    print(
        "Evaluation results saved to:"
    )

    print(
        f"  {output_path}"
    )

    print()
    print(
        "Evaluation completed successfully."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Evaluation interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print(
            "=" * 70
        )

        print(
            "EVALUATION FAILED"
        )

        print(
            "=" * 70
        )

        print(
            str(exc)
        )

        print()

        raise

