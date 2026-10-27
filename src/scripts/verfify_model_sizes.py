
from pathlib import Path

models_dir = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\Models"
)

model_files = [
    "CNN1D_keyword_spotting_fp32.tflite",
    "CNN1D_keyword_spotting_int8.tflite",
    "DSCNN_keyword_spotting_fp32.tflite",
    "DSCNN_keyword_spotting_int8.tflite",
    "mobilenetv2_fp32.tflite",
    "mobilenetv2_int8.tflite",
]

for filename in model_files:
    path = models_dir / filename

    if path.is_file():
        size_bytes = path.stat().st_size
        size_kb = size_bytes / 1024

        print(
            f"{filename}: "
            f"{size_bytes:,} bytes "
            f"({size_kb:.2f} KB)"
        )
    else:
        print(f"FILE NOT FOUND: {path}")
