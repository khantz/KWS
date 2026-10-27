import time
import numpy as np
import serial
from pathlib import Path

# -------- Config --------
PORT     = "COM3"       # <-- change to your ESP32 port
BAUD     = 921600
OUT_DIR  = Path(r"C:\Users\M S I\PycharmProjects\KWS\esp32_test")

FEATURES = np.load(OUT_DIR / "test_features_int8.npy")   # (N, 923)
LABELS   = np.load(OUT_DIR / "test_labels.npy")          # (N,) strings

CLASS_NAMES = ["no","off","on","one","stop","three","two","unknown","yes"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}

TRUE = np.array([CLASS_INDEX[c] for c in LABELS], dtype=np.int64)
N = FEATURES.shape[0]
print(f"Test samples : {N}")

# -------- Open serial --------
ser = serial.Serial(PORT, BAUD, timeout=10)
time.sleep(2)

# Wait for READY banner
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print("ESP32:", line)
    if line == "READY":
        break

# -------- Loop --------
correct = 0
latencies_us = []

t_start = time.time()

for i in range(N):
    sample = FEATURES[i].astype(np.int8).tobytes()
    n = len(sample).to_bytes(4, "little")

    ser.write(n)
    ser.write(sample)

    resp = ser.read(3)
    if len(resp) != 3:
        print(f"[{i}] no response")
        continue

    pred = resp[0]
    dt_us = (resp[1] | (resp[2] << 8)) * 100
    latencies_us.append(dt_us)

    if pred == TRUE[i]:
        correct += 1

    if (i + 1) % 100 == 0:
        acc = correct / (i + 1) * 100
        print(f"{i+1}/{N}   acc = {acc:.2f}%")

# -------- Final --------
acc = correct / N * 100
print()
print("=" * 55)
print("ESP32 INT8 INFERENCE ACCURACY")
print("=" * 55)
print(f"Samples       : {N}")
print(f"Correct       : {correct}")
print(f"Accuracy      : {acc:.2f}%")
print(f"Mean latency  : {np.mean(latencies_us):.0f} us")
print(f"Median latency: {np.median(latencies_us):.0f} us")
print(f"Elapsed       : {time.time() - t_start:.1f} s")
print("=" * 55)

# Save results
np.save(OUT_DIR / "esp32_latencies_us.npy", np.array(latencies_us))
with open(OUT_DIR / "esp32_accuracy.txt", "w") as f:
    f.write(f"Samples       : {N}\n")
    f.write(f"Correct       : {correct}\n")
    f.write(f"Accuracy      : {acc:.4f}%\n")
    f.write(f"Mean latency  : {np.mean(latencies_us):.0f} us\n")
    f.write(f"Median latency: {np.median(latencies_us):.0f} us\n")