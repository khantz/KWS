
from pathlib import Path
import os
import tensorflow as tf

models_dir = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\Models"
)

keras_models = {
    "CNN1D_keyword_spotting.keras":
        "CNN1D_keyword_spotting_fp32.tflite",

    "DSCNN_keyword_spotting.keras":
        "DSCNN_keyword_spotting_fp32.tflite",

    "MobileNetV2_best.keras":
        "mobilenetv2_fp32.tflite",
}

for keras_name, tflite_name in keras_models.items():
    keras_path = models_dir / keras_name
    output_path = models_dir / tflite_name
    temp_path = models_dir / f"{tflite_name}.tmp"

    print(f"\nProcessing: {keras_name}")

    if not keras_path.is_file():
        print(f"ERROR: Keras model not found: {keras_path}")
        continue

    try:
        # Load the trained model without requiring training configuration.
        model = tf.keras.models.load_model(
            str(keras_path),
            compile=False
        )

        # Convert to floating-point TensorFlow Lite.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        if not tflite_model:
            raise RuntimeError("Converter returned an empty model.")

        # Write to a temporary file first.
        temp_path.write_bytes(tflite_model)

        # Validate the converted model before replacing the output.
        interpreter = tf.lite.Interpreter(
            model_path=str(temp_path)
        )
        interpreter.allocate_tensors()

        input_shape = interpreter.get_input_details()[0]["shape"]
        output_shape = interpreter.get_output_details()[0]["shape"]

        # Replace the destination only after successful validation.
        os.replace(temp_path, output_path)

        size_bytes = output_path.stat().st_size

        print("SUCCESS")
        print(f"Output: {output_path.name}")
        print(f"Size: {size_bytes:,} bytes")
        print(f"Size: {size_bytes / 1024:.2f} KB")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")

    except Exception as exc:
        print(f"FAILED: {exc}")

    finally:
        if temp_path.exists():
            temp_path.unlink()

print("\nFP32 recovery process completed.")
