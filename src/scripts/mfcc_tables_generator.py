from pathlib import Path
import numpy as np

# ---------- Parameters (match your PC pipeline) ----------
SAMPLE_RATE = 16000
FFT_SIZE    = 512
N_MELS      = 26
N_MFCC      = 13
F_MIN       = 20.0
F_MAX       = 8000.0
FRAME_LEN   = 320

# ---------- Output path ----------
OUT_HEADER = Path(
    r"C:\Users\M S I\Documents\Arduino\Testheaderworkingornot\mfcc_tables.h"
)

# ---------- Mel helpers ----------
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

# ---------- Mel filter bank ----------
mel_min = hz_to_mel(F_MIN)
mel_max = hz_to_mel(F_MAX)
mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
hz_points  = mel_to_hz(mel_points)

frequency_bins = np.floor(
    (FFT_SIZE + 1) * hz_points / SAMPLE_RATE
).astype(int)

n_fft_bins = FFT_SIZE // 2 + 1   # 257

fb = np.zeros((N_MELS, n_fft_bins), dtype=np.float32)

for i in range(N_MELS):
    left   = frequency_bins[i]
    center = frequency_bins[i + 1]
    right  = frequency_bins[i + 2]

    if center > left:
        for b in range(left, center):
            fb[i, b] = (b - left) / (center - left)
    if right > center:
        for b in range(center, right):
            fb[i, b] = (right - b) / (right - center)

# ---------- DCT-II orthonormal ----------
dct_m = np.zeros((N_MFCC, N_MELS), dtype=np.float32)
for k in range(N_MFCC):
    ck = np.sqrt(1.0 / N_MELS) if k == 0 else np.sqrt(2.0 / N_MELS)
    for n in range(N_MELS):
        dct_m[k, n] = ck * np.cos(np.pi * k * (2 * n + 1) / (2 * N_MELS))

# ---------- Hamming window ----------
hamming = np.zeros(FRAME_LEN, dtype=np.float32)
for n in range(FRAME_LEN):
    hamming[n] = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (FRAME_LEN - 1))

# ---------- Write header ----------
def fmt2d(name, arr):
    lines = [f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
    for row in arr:
        lines.append("  {" + ", ".join(f"{v:.8e}f" for v in row) + "},")
    lines.append("};")
    return "\n".join(lines)

def fmt1d(name, arr):
    return (f"static const float {name}[{arr.shape[0]}] = {{\n  "
            + ", ".join(f"{v:.8e}f" for v in arr) + "\n};")

header = f"""// Auto-generated MFCC tables
// Matches PC pipeline: 20 ms frames, 30% overlap, Hamming window,
// 512-FFT, 26 Mel filters, natural log, DCT-II ortho, 13 coefficients.

#ifndef MFCC_TABLES_H_
#define MFCC_TABLES_H_

#define MFCC_SAMPLE_RATE    {SAMPLE_RATE}
#define MFCC_FFT_SIZE       {FFT_SIZE}
#define MFCC_FRAME_LEN      {FRAME_LEN}
#define MFCC_FRAME_HOP      224
#define MFCC_N_FRAMES       71
#define MFCC_N_MELS         {N_MELS}
#define MFCC_N_MFCC         {N_MFCC}
#define MFCC_N_FFT_BINS     {n_fft_bins}
#define MFCC_PREEMPH        0.97f
#define MFCC_LOG_EPSILON    1.0e-10f
#define MFCC_FFT_SCALE      (1.0f / {FFT_SIZE}.0f)

{fmt2d("mel_filterbank", fb)}

{fmt2d("dct_matrix", dct_m)}

{fmt1d("hamming_window", hamming)}

#endif
"""

OUT_HEADER.parent.mkdir(parents=True, exist_ok=True)
OUT_HEADER.write_text(header, encoding="utf-8")

print(f"Wrote: {OUT_HEADER}")
print(f"Mel filter bank : {fb.shape}")
print(f"DCT matrix      : {dct_m.shape}")
print(f"Hamming window  : {hamming.shape}")