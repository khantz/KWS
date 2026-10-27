
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import dct

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
from src.feature_extraction.mfcc_extractor import (
    MFCCFeatureExtractor,
)

# =========================================================
# 2. Parameters matching your MFCC implementation
# =========================================================

SAMPLE_RATE = 16000
FFT_SIZE = 512
FRAME_DURATION_MS = 20.0
FRAME_OVERLAP_RATIO = 0.30
NUM_MEL_FILTERS = 26
NUM_MFCC = 13
PRE_EMPHASIS_COEFFICIENT = 0.97
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
    "grid.alpha": 0.35,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 600,
})

# Perceptually uniform scientific palette
FEATURE_CMAP = "viridis"
FILTER_CMAP = plt.get_cmap("turbo")

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
        "Use the same resampling pipeline as your training code."
    )

audio = np.nan_to_num(audio)

if audio.size == 0:
    raise ValueError("The input audio file is empty.")

duration = len(audio) / SAMPLE_RATE

# =========================================================
# 5. Initialize feature extraction classes
# =========================================================

signal_processor = SignalProcessor(
    sample_rate=SAMPLE_RATE,
    frame_duration_ms=FRAME_DURATION_MS,
    frame_overlap_ratio=FRAME_OVERLAP_RATIO,
    window_type="hamming",
    fft_size=FFT_SIZE,
    pre_emphasis_enabled=True,
    pre_emphasis_coefficient=PRE_EMPHASIS_COEFFICIENT,
)

mel_filter_bank = MelFilterBank(
    sample_rate=SAMPLE_RATE,
    fft_size=FFT_SIZE,
    number_of_filters=NUM_MEL_FILTERS,
    minimum_frequency=20.0,
    maximum_frequency=8000.0,
)

mfcc_extractor = MFCCFeatureExtractor(
    sample_rate=SAMPLE_RATE,
    pre_emphasis_enabled=True,
    pre_emphasis_coefficient=PRE_EMPHASIS_COEFFICIENT,
    frame_duration_ms=FRAME_DURATION_MS,
    frame_overlap_ratio=FRAME_OVERLAP_RATIO,
    window_type="hamming",
    fft_size=FFT_SIZE,
    number_of_mel_filters=NUM_MEL_FILTERS,
    minimum_frequency=20.0,
    maximum_frequency=8000.0,
    log_epsilon=LOG_EPSILON,
    dct_type=2,
    dct_normalization="ortho",
    number_of_mfcc_coefficients=NUM_MFCC,
)

# =========================================================
# 6. Calculate intermediate feature representations
# =========================================================

# Shape: (frames, 257 frequency bins)
power_spectrum = signal_processor.process(audio)

# Shape: (frames, 26 Mel filters)
mel_energies = mel_filter_bank.apply(power_spectrum)

# Logarithmic compression
log_mel_energies = np.log(
    mel_energies + LOG_EPSILON
).astype(np.float32)

# DCT-II across the Mel-filter dimension
all_dct_coefficients = dct(
    log_mel_energies,
    type=2,
    axis=1,
    norm="ortho",
)

# Retain the first 13 coefficients
mfcc = all_dct_coefficients[:, :NUM_MFCC].astype(
    np.float32
)

# Confirm consistency with the actual extractor
mfcc_check = mfcc_extractor.extract(audio)

if not np.allclose(
    mfcc,
    mfcc_check,
    rtol=1e-4,
    atol=1e-5,
):
    raise ValueError(
        "Calculated MFCC values do not match "
        "MFCCFeatureExtractor.extract()."
    )

# Time coordinates for frame-based features
frame_length = int(
    SAMPLE_RATE * FRAME_DURATION_MS / 1000
)
hop_length = int(
    frame_length * (1.0 - FRAME_OVERLAP_RATIO)
)

num_frames = power_spectrum.shape[0]

frame_times = (
    np.arange(num_frames) * hop_length
    + frame_length / 2
) / SAMPLE_RATE

print(f"Audio duration: {duration:.3f} s")
print(f"Power spectrum: {power_spectrum.shape}")
print(f"Mel energies: {mel_energies.shape}")
print(f"Log-Mel energies: {log_mel_energies.shape}")
print(f"Full DCT coefficients: {all_dct_coefficients.shape}")
print(f"Final MFCC features: {mfcc.shape}")

# =========================================================
# 7. Figure 1: MFCC feature extraction
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
    constrained_layout=True,
)

fig.suptitle(
    "Mel-Frequency Cepstral Coefficient (MFCC) Extraction",
    fontsize=14,
    fontweight="bold",
)

# ---------------------------------------------------------
# (a) Log-Mel filter-bank energies
# ---------------------------------------------------------

ax = axes[0, 0]

mesh_a = ax.pcolormesh(
    frame_times,
    np.arange(1, NUM_MEL_FILTERS + 1),
    log_mel_energies.T,
    shading="auto",
    cmap=FEATURE_CMAP,
)

ax.set_title("(a) Log-Mel Filter-Bank Energies", loc="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mel Filter Index")
ax.set_yticks([1, 5, 10, 15, 20, 26])

cbar_a = fig.colorbar(mesh_a, ax=ax, pad=0.025)
cbar_a.set_label("Log Energy")

# ---------------------------------------------------------
# (b) DCT coefficients before truncation
# ---------------------------------------------------------

ax = axes[0, 1]

mesh_b = ax.pcolormesh(
    frame_times,
    np.arange(1, NUM_MEL_FILTERS + 1),
    all_dct_coefficients.T,
    shading="auto",
    cmap=FEATURE_CMAP,
)

ax.set_title("(b) DCT Coefficients Before Truncation", loc="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("DCT Coefficient Index")
ax.set_yticks([1, 5, 10, 15, 20, 26])

cbar_b = fig.colorbar(mesh_b, ax=ax, pad=0.025)
cbar_b.set_label("Coefficient Value")

# ---------------------------------------------------------
# (c) Final 13 MFCC coefficients
# ---------------------------------------------------------

ax = axes[1, 0]

mesh_c = ax.pcolormesh(
    frame_times,
    np.arange(NUM_MFCC),
    mfcc.T,
    shading="auto",
    cmap=FEATURE_CMAP,
)

ax.set_title("(c) Retained 13 MFCC Coefficients", loc="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("MFCC Coefficient Index")
ax.set_yticks(np.arange(NUM_MFCC))

cbar_c = fig.colorbar(mesh_c, ax=ax, pad=0.025)
cbar_c.set_label("MFCC Value")

# ---------------------------------------------------------
# (d) MFCC vector for a representative frame
# ---------------------------------------------------------

ax = axes[1, 1]

representative_frame = num_frames // 2
coefficient_indices = np.arange(NUM_MFCC)

bar_colors = plt.get_cmap(FEATURE_CMAP)(
    np.linspace(0.2, 0.85, NUM_MFCC)
)

ax.bar(
    coefficient_indices,
    mfcc[representative_frame],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.45,
    width=0.78,
)

ax.axhline(
    0,
    color="black",
    linewidth=0.8,
)

ax.set_title(
    f"(d) MFCC Vector at Frame {representative_frame}",
    loc="left",
)
ax.set_xlabel("MFCC Coefficient Index")
ax.set_ylabel("Coefficient Value")
ax.set_xticks(coefficient_indices)
ax.grid(True, axis="y")
ax.grid(False, axis="x")

# Save the figure
pipeline_png = OUTPUT_DIR / "mfcc_feature_pipeline.png"
pipeline_pdf = OUTPUT_DIR / "mfcc_feature_pipeline.pdf"

fig.savefig(
    pipeline_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    pipeline_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

# =========================================================
# 8. Figure 2: Triangular Mel filter-bank responses
# =========================================================

filters = mel_filter_bank.filters

frequencies = np.fft.rfftfreq(
    FFT_SIZE,
    d=1.0 / SAMPLE_RATE,
)

fig, ax = plt.subplots(figsize=(10, 5.5))

for index, filt in enumerate(filters):
    ax.plot(
        frequencies,
        filt,
        color=FILTER_CMAP(index / (NUM_MEL_FILTERS - 1)),
        linewidth=0.9,
        alpha=0.9,
    )

ax.set_title(
    "Triangular Mel Filter-Bank Responses",
    loc="left",
    fontweight="bold",
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Filter Weight")
ax.set_xlim(0, SAMPLE_RATE / 2)
ax.set_ylim(0, 1.05)
ax.grid(True)

fig.tight_layout()

filters_png = OUTPUT_DIR / "mfcc_mel_filter_bank.png"
filters_pdf = OUTPUT_DIR / "mfcc_mel_filter_bank.pdf"

fig.savefig(
    filters_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    filters_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

# =========================================================
# 9. Report output paths
# =========================================================

print("\nFigures saved successfully:")
print(pipeline_png)
print(pipeline_pdf)
print(filters_png)
print(filters_pdf)
