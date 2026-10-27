
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# =========================================================
# 1. Project configuration
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\M S I\PycharmProjects\KWS"
)

AUDIO_PATH = PROJECT_ROOT / (
    r"dataset\after_preprocessed\test\off"
    r"\0a2b400e_nohash_3.wav"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tflite_evaluation"
    / "plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.signal_processor import (
    SignalProcessor,
)
from src.feature_extraction.mel_filter_bank import (
    MelFilterBank,
)
from src.feature_extraction.mel_spectrogram_extractor import (
    MelSpectrogramFeatureExtractor,
)

# =========================================================
# 2. Parameters matching your implementation
# =========================================================

SAMPLE_RATE = 16000
FFT_SIZE = 512
FRAME_DURATION_MS = 20.0
FRAME_OVERLAP_RATIO = 0.30
NUM_MEL_FILTERS = 40
LOG_EPSILON = 1.0e-10

# =========================================================
# 3. Academic plotting style
# =========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 600,
})

FEATURE_CMAP = "magma"

# =========================================================
# 4. Load audio
# =========================================================

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_PATH}"
    )

original_sr, raw_audio = wavfile.read(AUDIO_PATH)

if raw_audio.ndim == 2:
    raw_audio = raw_audio.astype(np.float32).mean(axis=1)

if np.issubdtype(raw_audio.dtype, np.integer):
    dtype_info = np.iinfo(raw_audio.dtype)
    scale = float(max(abs(dtype_info.min), abs(dtype_info.max)))
    audio = raw_audio.astype(np.float32) / scale
else:
    audio = raw_audio.astype(np.float32)

if original_sr != SAMPLE_RATE:
    raise ValueError(
        f"Expected {SAMPLE_RATE} Hz audio, got {original_sr} Hz. "
        "Apply the same resampling procedure used during training."
    )

audio = np.nan_to_num(audio)

if audio.size == 0:
    raise ValueError("The WAV file is empty.")

duration = len(audio) / SAMPLE_RATE

# =========================================================
# 5. Initialize the feature extractor
# =========================================================

extractor = MelSpectrogramFeatureExtractor(
    sample_rate=SAMPLE_RATE,
    pre_emphasis_enabled=False,
    pre_emphasis_coefficient=0.97,
    frame_duration_ms=FRAME_DURATION_MS,
    frame_overlap_ratio=FRAME_OVERLAP_RATIO,
    window_type="hamming",
    fft_size=FFT_SIZE,
    number_of_mel_filters=NUM_MEL_FILTERS,
    minimum_frequency=20.0,
    maximum_frequency=8000.0,
    log_epsilon=LOG_EPSILON,
)

# Use the actual implementation to extract features.
log_mel = extractor.extract(audio)

if log_mel.ndim != 2:
    raise ValueError(
        "Expected a 2D log-Mel array with shape "
        "(frames, Mel filters); received "
        f"{log_mel.shape}"
    )

num_frames, num_filters = log_mel.shape

if num_filters != NUM_MEL_FILTERS:
    raise ValueError(
        f"Expected {NUM_MEL_FILTERS} Mel filters, "
        f"received {num_filters}"
    )

# Match the framing configuration for the time axis.
frame_length = int(
    SAMPLE_RATE * FRAME_DURATION_MS / 1000
)

hop_length = int(
    frame_length * (1.0 - FRAME_OVERLAP_RATIO)
)

frame_times = (
    np.arange(num_frames) * hop_length
    + frame_length / 2
) / SAMPLE_RATE

mel_filter_indices = np.arange(1, NUM_MEL_FILTERS + 1)

print(f"Sample rate: {original_sr} Hz")
print(f"Audio duration: {duration:.3f} seconds")
print(f"Log-Mel feature shape: {log_mel.shape}")

# =========================================================
# 6. Calculate summary statistics
# =========================================================

# Average energy per Mel filter over all frames.
mean_energy_per_filter = np.mean(log_mel, axis=0)

# Average energy per frame across all Mel filters.
mean_energy_per_frame = np.mean(log_mel, axis=1)

# =========================================================
# 7. Create the academic figure
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
    constrained_layout=True,
    gridspec_kw={
        "height_ratios": [1.15, 1.0],
    },
)

fig.suptitle(
    "Log-Mel Spectrogram Feature Analysis",
    fontsize=14,
    fontweight="bold",
)

# ---------------------------------------------------------
# (a) Log-Mel spectrogram
# ---------------------------------------------------------

ax = axes[0, 0]

image = ax.pcolormesh(
    frame_times,
    mel_filter_indices,
    log_mel.T,
    shading="auto",
    cmap=FEATURE_CMAP,
)

ax.set_title("(a) Log-Mel Spectrogram", loc="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mel Filter Index")
ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35, 40])

colorbar = fig.colorbar(image, ax=ax, pad=0.025)
colorbar.set_label("Log Energy")

# ---------------------------------------------------------
# (b) Mean log-Mel energy across filters
# ---------------------------------------------------------

ax = axes[0, 1]

ax.plot(
    frame_times,
    mean_energy_per_frame,
    color="#31688E",
    linewidth=1.1,
)

ax.fill_between(
    frame_times,
    mean_energy_per_frame,
    np.min(mean_energy_per_frame),
    color="#35B779",
    alpha=0.22,
)

ax.set_title(
    "(b) Mean Log-Mel Energy Across Time",
    loc="left",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean Log Energy")

# ---------------------------------------------------------
# (c) Mean log-Mel energy per filter
# ---------------------------------------------------------

ax = axes[1, 0]

filter_colors = plt.get_cmap(FEATURE_CMAP)(
    np.linspace(0.15, 0.9, NUM_MEL_FILTERS)
)

ax.bar(
    mel_filter_indices,
    mean_energy_per_filter,
    color=filter_colors,
    edgecolor="black",
    linewidth=0.3,
    width=0.82,
)

ax.set_title(
    "(c) Mean Log-Mel Energy by Filter",
    loc="left",
)
ax.set_xlabel("Mel Filter Index")
ax.set_ylabel("Mean Log Energy")
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax.grid(True, axis="y")
ax.grid(False, axis="x")

# ---------------------------------------------------------
# (d) Distribution of log-Mel values
# ---------------------------------------------------------

ax = axes[1, 1]

ax.hist(
    log_mel.ravel(),
    bins=45,
    color="#31688E",
    edgecolor="white",
    linewidth=0.4,
    alpha=0.95,
)

ax.set_title(
    "(d) Distribution of Log-Mel Feature Values",
    loc="left",
)
ax.set_xlabel("Log Energy")
ax.set_ylabel("Frequency")
ax.grid(True, axis="y")
ax.grid(False, axis="x")

# =========================================================
# 8. Save PNG and PDF
# =========================================================

png_path = OUTPUT_DIR / "log_mel_spectrogram_analysis.png"
pdf_path = OUTPUT_DIR / "log_mel_spectrogram_analysis.pdf"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("\nFigures saved successfully:")
print(png_path)
print(pdf_path)
