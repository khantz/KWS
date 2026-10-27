
from pathlib import Path

model_dir = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\Models"
)

models = [
    "DSCNN_keyword_spotting_int8.tflite",
    "CNN1D_keyword_spotting_int8.tflite",
]

for filename in models:
    path = model_dir / filename

    if not path.exists():
        print(f"MISSING: {filename}")
    else:
        print(f"{filename}: {path.stat().st_size:,} bytes")
