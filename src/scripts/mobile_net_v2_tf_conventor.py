from pathlib import Path
import random

import numpy as np
import tensorflow as tf


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\M S I\PycharmProjects\KWS"
)

MODELS_DIR = PROJECT_ROOT / "Models"

# ------------------------------------------------------------
# Trained MobileNetV2 Keras model
# ------------------------------------------------------------

MODEL_PATH = (
    MODELS_DIR
    / "MobileNetV2_keyword_spotting.keras"
)

# ------------------------------------------------------------
# MobileNetV2 normalization parameters
# ------------------------------------------------------------

NORMALIZATION_PATH = (
    MODELS_DIR
    / "normalization_parameters"
    / "MobileNetV2_normalization.npz"
)

# ------------------------------------------------------------
# MobileNetV2 feature directory
# ------------------------------------------------------------

FEATURE_DIR = (
    PROJECT_ROOT
    / "feature"
    / "mel_spectrogram"
)

# ------------------------------------------------------------
# TFLite output files
# ------------------------------------------------------------

FP32_OUTPUT_PATH = (
    MODELS_DIR
    / "mobilenetv2_fp32.tflite"
)

INT8_OUTPUT_PATH = (
    MODELS_DIR
    / "mobilenetv2_int8.tflite"
)


# ============================================================
# INT8 CALIBRATION SETTINGS
# ============================================================

# Number of samples used by TensorFlow for calibration.
REPRESENTATIVE_SAMPLES = 300

# Reproducibility
RANDOM_SEED = 42


# ============================================================
# UTILITY
# ============================================================

def print_separator():
    print("=" * 70)


def print_file_size(path: Path):

    size_bytes = path.stat().st_size
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024

    print(f"File : {path}")
    print(f"Size : {size_bytes:,} bytes")
    print(f"     : {size_kb:.2f} KB")
    print(f"     : {size_mb:.2f} MB")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_paths():

    print_separator()
    print("Checking required files and directories")
    print_separator()

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nModels directory:")
    print(MODELS_DIR)

    print("\nKeras model:")
    print(MODEL_PATH)

    print("\nNormalization file:")
    print(NORMALIZATION_PATH)

    print("\nFeature directory:")
    print(FEATURE_DIR)

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"\nProject root does not exist:\n"
            f"{PROJECT_ROOT}"
        )

    # --------------------------------------------------------
    # Models directory
    # --------------------------------------------------------

    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"\nModels directory does not exist:\n"
            f"{MODELS_DIR}"
        )

    # --------------------------------------------------------
    # Keras model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nKeras model not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"\nNormalization file not found:\n"
            f"{NORMALIZATION_PATH}"
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"\nFeature directory not found:\n"
            f"{FEATURE_DIR}"
        )

    print("\nAll required paths exist.")


# ============================================================
# LOAD NORMALIZATION PARAMETERS
# ============================================================

def load_normalization_parameters():

    print_separator()
    print("Loading normalization parameters")
    print_separator()

    data = np.load(
        NORMALIZATION_PATH
    )

    print("\nKeys found in normalization file:")

    for key in data.files:
        print(f"  - {key}")

    if "minimum" not in data.files:
        raise KeyError(
            "Normalization file does not contain "
            "'minimum'."
        )

    if "maximum" not in data.files:
        raise KeyError(
            "Normalization file does not contain "
            "'maximum'."
        )

    train_min = float(
        data["minimum"]
    )

    train_max = float(
        data["maximum"]
    )

    print(
        f"\nTraining minimum : {train_min}"
    )

    print(
        f"Training maximum : {train_max}"
    )

    if train_max == train_min:
        raise ValueError(
            "Training minimum and maximum are identical."
        )

    return train_min, train_max


# ============================================================
# LOAD KERAS MODEL
# ============================================================

def load_keras_model():

    print_separator()
    print("Loading MobileNetV2 Keras model")
    print_separator()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nModel loaded successfully.")

    print(
        f"Input shape  : {model.input_shape}"
    )

    print(
        f"Output shape : {model.output_shape}"
    )

    return model


# ============================================================
# MOBILE NET V2 PREPROCESSING
# ============================================================

def convert_to_mobilenet_input(
    features,
    train_min,
    train_max
):
    """
    Reproduce the preprocessing used during
    MobileNetV2 training.

    Original feature:
        (frames, mel_bins)

    Processing:
        1. Convert to float32
        2. Add channel dimension
        3. Resize to 96 x 96
        4. Convert 1 channel to 3 channels
        5. Normalize using training min/max
        6. Scale to approximately [-1, 1]
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Add channel dimension
    # --------------------------------------------------------

    if features.ndim == 2:

        features = features[
            ...,
            np.newaxis
        ]

    elif features.ndim == 3:

        pass

    else:

        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}"
        )

    # --------------------------------------------------------
    # Convert NumPy -> TensorFlow tensor
    # --------------------------------------------------------

    tensor = tf.convert_to_tensor(
        features,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    tensor = tf.image.resize(
        tensor,
        [96, 96],
        method="bilinear"
    )

    # --------------------------------------------------------
    # Convert grayscale -> RGB
    # --------------------------------------------------------

    channels = tensor.shape[-1]

    if channels == 1:

        tensor = tf.repeat(
            tensor,
            repeats=3,
            axis=-1
        )

    elif channels == 3:

        pass

    else:

        raise ValueError(
            f"Unexpected number of channels: "
            f"{channels}"
        )

    # --------------------------------------------------------
    # Normalize exactly as training
    #
    # normalized =
    #
    # 2 * (x - min)
    # ---------------- - 1
    #      max - min
    #
    # --------------------------------------------------------

    tensor = (
        2.0
        * (tensor - train_min)
        / (train_max - train_min)
        - 1.0
    )

    return tensor.numpy().astype(
        np.float32
    )


# ============================================================
# FIND FEATURE FILES
# ============================================================

def find_feature_files():

    print_separator()
    print("Searching for feature files")
    print_separator()

    feature_files = sorted(
        FEATURE_DIR.rglob("*.npy")
    )

    if not feature_files:

        raise FileNotFoundError(
            f"\nNo .npy files found in:\n"
            f"{FEATURE_DIR}"
        )

    print(
        f"\nFound "
        f"{len(feature_files):,} feature files."
    )

    # --------------------------------------------------------
    # Show first few files
    # --------------------------------------------------------

    print("\nExample feature files:")

    for path in feature_files[:5]:

        print(
            f"  {path.relative_to(FEATURE_DIR)}"
        )

    return feature_files


# ============================================================
# REPRESENTATIVE DATASET
# ============================================================

def representative_dataset(
    feature_files,
    train_min,
    train_max
):
    """
    Generator used for INT8 calibration.

    The data must go through the SAME preprocessing
    pipeline used during MobileNetV2 training.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    files = list(
        feature_files
    )

    rng.shuffle(files)

    selected_files = files[
        :min(
            REPRESENTATIVE_SAMPLES,
            len(files)
        )
    ]

    print_separator()
    print("Preparing representative dataset")
    print_separator()

    print(
        f"Available samples : "
        f"{len(files):,}"
    )

    print(
        f"Calibration samples : "
        f"{len(selected_files):,}"
    )

    for index, feature_path in enumerate(
        selected_files,
        start=1
    ):

        # ----------------------------------------------------
        # Load feature
        # ----------------------------------------------------

        features = np.load(
            feature_path
        )

        # ----------------------------------------------------
        # Apply exact MobileNetV2 preprocessing
        # ----------------------------------------------------

        processed = convert_to_mobilenet_input(
            features,
            train_min,
            train_max
        )

        # ----------------------------------------------------
        # Add batch dimension
        #
        # (96, 96, 3)
        #       ↓
        # (1, 96, 96, 3)
        # ----------------------------------------------------

        processed = np.expand_dims(
            processed,
            axis=0
        )

        processed = processed.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Print first sample information
        # ----------------------------------------------------

        if index == 1:

            print("\nFirst representative sample:")

            print(
                f"File : {feature_path}"
            )

            print(
                f"Shape: {processed.shape}"
            )

            print(
                f"Dtype: {processed.dtype}"
            )

            print(
                f"Min  : {processed.min():.6f}"
            )

            print(
                f"Max  : {processed.max():.6f}"
            )

        yield [
            processed
        ]


# ============================================================
# CONVERT FP32 TFLITE
# ============================================================

def convert_fp32(model):

    print_separator()
    print("FP32 TFLite Conversion")
    print_separator()

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    # No quantization
    tflite_model = converter.convert()

    with open(
        FP32_OUTPUT_PATH,
        "wb"
    ) as file:

        file.write(
            tflite_model
        )

    print(
        "\nFP32 conversion successful."
    )

    print_file_size(
        FP32_OUTPUT_PATH
    )


# ============================================================
# CONVERT FULL INT8 TFLITE
# ============================================================

def convert_int8(
    model,
    feature_files,
    train_min,
    train_max
):

    print_separator()
    print("FULL INT8 TFLite Conversion")
    print_separator()

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    # --------------------------------------------------------
    # Enable post-training quantization
    # --------------------------------------------------------

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # --------------------------------------------------------
    # Representative dataset
    # --------------------------------------------------------

    converter.representative_dataset = (
        lambda:
        representative_dataset(
            feature_files,
            train_min,
            train_max
        )
    )

    # --------------------------------------------------------
    # Require INT8 operations
    # --------------------------------------------------------

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # --------------------------------------------------------
    # Force INT8 input/output
    # --------------------------------------------------------

    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    print(
        "\nStarting full INT8 conversion..."
    )

    print(
        "TensorFlow will use the "
        "representative dataset for calibration."
    )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    tflite_model = converter.convert()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        INT8_OUTPUT_PATH,
        "wb"
    ) as file:

        file.write(
            tflite_model
        )

    print(
        "\nINT8 conversion successful."
    )

    print_file_size(
        INT8_OUTPUT_PATH
    )


# ============================================================
# INSPECT TFLITE MODEL
# ============================================================

def inspect_tflite_model(
    model_path,
    model_name
):

    print_separator()
    print(
        f"Inspecting {model_name}"
    )
    print_separator()

    interpreter = tf.lite.Interpreter(
        model_path=str(
            model_path
        )
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter
        .get_input_details()
    )

    output_details = (
        interpreter
        .get_output_details()
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    print("\nINPUT")

    for detail in input_details:

        print(
            f"Name       : "
            f"{detail['name']}"
        )

        print(
            f"Shape      : "
            f"{detail['shape']}"
        )

        print(
            f"Dtype      : "
            f"{detail['dtype']}"
        )

        print(
            f"Quant      : "
            f"{detail['quantization']}"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\nOUTPUT")

    for detail in output_details:

        print(
            f"Name       : "
            f"{detail['name']}"
        )

        print(
            f"Shape      : "
            f"{detail['shape']}"
        )

        print(
            f"Dtype      : "
            f"{detail['dtype']}"
        )

        print(
            f"Quant      : "
            f"{detail['quantization']}"
        )


# ============================================================
# VALIDATE INT8 MODEL
# ============================================================

def validate_int8_model():

    print_separator()
    print("Validating INT8 model")
    print_separator()

    interpreter = tf.lite.Interpreter(
        model_path=str(
            INT8_OUTPUT_PATH
        )
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter
        .get_input_details()
    )

    output_details = (
        interpreter
        .get_output_details()
    )

    input_detail = input_details[0]
    output_detail = output_details[0]

    input_dtype = (
        input_detail["dtype"]
    )

    output_dtype = (
        output_detail["dtype"]
    )

    input_quantization = (
        input_detail["quantization"]
    )

    output_quantization = (
        output_detail["quantization"]
    )

    print(
        f"Input dtype       : "
        f"{input_dtype}"
    )

    print(
        f"Output dtype      : "
        f"{output_dtype}"
    )

    print(
        f"Input scale       : "
        f"{input_quantization[0]}"
    )

    print(
        f"Input zero point  : "
        f"{input_quantization[1]}"
    )

    print(
        f"Output scale      : "
        f"{output_quantization[0]}"
    )

    print(
        f"Output zero point : "
        f"{output_quantization[1]}"
    )

    # --------------------------------------------------------
    # Check INT8 input
    # --------------------------------------------------------

    if input_dtype != np.int8:

        raise RuntimeError(
            "\nINT8 validation failed.\n"
            f"Expected input dtype: {np.int8}\n"
            f"Actual input dtype  : {input_dtype}"
        )

    # --------------------------------------------------------
    # Check INT8 output
    # --------------------------------------------------------

    if output_dtype != np.int8:

        raise RuntimeError(
            "\nINT8 validation failed.\n"
            f"Expected output dtype: {np.int8}\n"
            f"Actual output dtype  : {output_dtype}"
        )

    print(
        "\nINT8 validation PASSED."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_separator()
    print("MobileNetV2 TFLite Converter")
    print_separator()

    # --------------------------------------------------------
    # 1. Check paths
    # --------------------------------------------------------

    check_required_paths()

    # --------------------------------------------------------
    # 2. Load normalization parameters
    # --------------------------------------------------------

    train_min, train_max = (
        load_normalization_parameters()
    )

    # --------------------------------------------------------
    # 3. Load trained Keras model
    # --------------------------------------------------------

    model = load_keras_model()

    # --------------------------------------------------------
    # 4. Find feature files
    # --------------------------------------------------------

    feature_files = (
        find_feature_files()
    )

    # --------------------------------------------------------
    # 5. Convert FP32
    # --------------------------------------------------------

    convert_fp32(
        model
    )

    # --------------------------------------------------------
    # 6. Convert full INT8
    # --------------------------------------------------------

    convert_int8(
        model,
        feature_files,
        train_min,
        train_max
    )

    # --------------------------------------------------------
    # 7. Inspect FP32
    # --------------------------------------------------------

    inspect_tflite_model(
        FP32_OUTPUT_PATH,
        "FP32 TFLite"
    )

    # --------------------------------------------------------
    # 8. Inspect INT8
    # --------------------------------------------------------

    inspect_tflite_model(
        INT8_OUTPUT_PATH,
        "INT8 TFLite"
    )

    # --------------------------------------------------------
    # 9. Validate INT8
    # --------------------------------------------------------

    validate_int8_model()

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print()
    print_separator()
    print("CONVERSION COMPLETE")
    print_separator()

    print("\nGenerated files:")

    print(
        f"\nFP32 TFLite:"
        f"\n{FP32_OUTPUT_PATH}"
    )

    print(
        f"\nINT8 TFLite:"
        f"\n{INT8_OUTPUT_PATH}"
    )

    print("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()