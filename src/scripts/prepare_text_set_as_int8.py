import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL   = r"C:\Users\M S I\PycharmProjects\KWS\Models\CNN1D_keyword_spotting_int8.tflite"
FEATDIR = Path(r"C:\Users\M S I\PycharmProjects\KWS\feature\mfcc_feature\test")
OUT_DIR = Path(r"C:\Users\M S I\PycharmProjects\KWS\esp32_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model to get input scale / zero point
interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
scale = float(inp["quantization"][0])
zp    = int(inp["quantization"][1])
print(f"Input scale = {scale}")
print(f"Input zero  = {zp}")

# Load all test MFCC features
files = sorted(FEATDIR.rglob("*.npy"))
print(f"Found {len(files)} test files")

X_int8 = []
for f in files:
    arr = np.load(f).astype(np.float32)
    if arr.shape == (13, 71):
        arr = arr.T
    assert arr.shape == (71, 13), f"{f}: {arr.shape}"
    X_int8.append(arr.reshape(-1))

X_int8 = np.stack(X_int8, axis=0)   # (N, 923)
N = X_int8.shape[0]

# Quantize: q = round(x / scale) + zp, clipped to [-128, 127]
Q = np.round(X_int8 / scale + zp).astype(np.int32)
Q = np.clip(Q, -128, 127).astype(np.int8)

np.save(OUT_DIR / "test_features_int8.npy", Q)
print(f"Saved {OUT_DIR / 'test_features_int8.npy'}  shape={Q.shape}")