
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.io import wavfile


# ============================================================
# 1. FILE PATHS
# ============================================================

AUDIO_PATH = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\dataset\after_preprocessed\test\off\0a2b400e_nohash_3.wav"
)

OUTPUT_DIR = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "signal_processing_pipeline.png"
OUTPUT_PDF = OUTPUT_DIR / "signal_processing_pipeline.pdf"


# ============================================================
# 2. SIGNAL PROCESSING PARAMETERS
# Match the supplied SignalProcessor class
# ============================================================

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20.0
FRAME_OVERLAP_RATIO = 0.30
WINDOW_TYPE = "hamming"
FFT_SIZE = 512

PRE_EMPHASIS_ENABLED = False
PRE_EMPHASIS_COEFFICIENT = 0.97

FRAME_LENGTH = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_OVERLAP = int(FRAME_LENGTH * FRAME_OVERLAP_RATIO)
HOP_LENGTH = FRAME_LENGTH - FRAME_OVERLAP

# Use distinct shades from one blue palette.
BLUE = {
    "navy": "#17365D",
    "dark": "#245A91",
    "medium": "#357ABD",
    "bright": "#4F9DDB",
    "light": "#86BCE8",
    "pale": "#C9E0F5",
    "grid": "#D8E2ED",
    "text": "#263746",
}


# ============================================================
# 3. PUBLICATION-STYLE MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.edgecolor": BLUE["grid"],
    "axes.linewidth": 0.8,
    "axes.labelcolor": BLUE["text"],
    "xtick.color": BLUE["text"],
    "ytick.color": BLUE["text"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 4. LOAD AND VALIDATE THE AUDIO
# ============================================================

if not AUDIO_PATH.is_file():
    raise FileNotFoundError(f"Audio file not found:\n{AUDIO_PATH}")

sample_rate, raw_audio = wavfile.read(AUDIO_PATH)

if sample_rate != SAMPLE_RATE:
    raise ValueError(
        f"Expected {SAMPLE_RATE} Hz, received {sample_rate} Hz. "
        "Resample the WAV file before processing."
    )

original_dtype = raw_audio.dtype

if raw_audio.ndim == 2:
    raw_audio = raw_audio.astype(np.float64).mean(axis=1)
else:
    raw_audio = raw_audio.astype(np.float64)

# Convert integer PCM to approximately [-1, 1].
if np.issubdtype(original_dtype, np.integer):
    info = np.iinfo(original_dtype)
    pcm_scale = float(max(abs(info.min), abs(info.max)))
    audio = (raw_audio / pcm_scale).astype(np.float32)
else:
    audio = raw_audio.astype(np.float32)

if audio.size < FRAME_LENGTH:
    raise ValueError(
        "The audio contains fewer samples than one analysis frame."
    )

duration = len(audio) / SAMPLE_RATE
time_seconds = np.arange(len(audio)) / SAMPLE_RATE


# ============================================================
# 5. PRE-EMPHASIS
# ============================================================

if PRE_EMPHASIS_ENABLED:
    processed_audio = np.empty_like(audio, dtype=np.float32)
    processed_audio[0] = audio[0]
    processed_audio[1:] = (
        audio[1:]
        - PRE_EMPHASIS_COEFFICIENT * audio[:-1]
    )
else:
    processed_audio = audio.copy()


# ============================================================
# 6. FRAME SEGMENTATION
# ============================================================

number_of_frames = (
    1 + (len(processed_audio) - FRAME_LENGTH) // HOP_LENGTH
)

frames = np.empty(
    (number_of_frames, FRAME_LENGTH),
    dtype=np.float32,
)

for index in range(number_of_frames):
    start = index * HOP_LENGTH
    frames[index] = processed_audio[
        start:start + FRAME_LENGTH
    ]

# The first frame is used for the windowing and FFT illustrations.
selected_frame = frames[0]

frame_time_ms = (
    np.arange(FRAME_LENGTH) / SAMPLE_RATE * 1000
)


# ============================================================
# 7. HAMMING WINDOW AND WINDOWED FRAME
# ============================================================

if WINDOW_TYPE.lower() != "hamming":
    raise ValueError(f"Unsupported window: {WINDOW_TYPE}")

window = np.hamming(FRAME_LENGTH).astype(np.float32)
windowed_frame = selected_frame * window


# ============================================================
# 8. POWER SPECTRUM
# ============================================================

fft_result = np.fft.rfft(
    windowed_frame,
    n=FFT_SIZE,
)

power_spectrum = np.abs(fft_result) ** 2 / FFT_SIZE

frequencies = np.fft.rfftfreq(
    FFT_SIZE,
    d=1 / SAMPLE_RATE,
)


# ============================================================
# 9. CONSISTENT AXIS FORMATTING
# ============================================================

def format_axis(ax):
    ax.grid(
        True,
        color=BLUE["grid"],
        linestyle=":",
        linewidth=0.7,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLUE["grid"])
    ax.spines["bottom"].set_color(BLUE["grid"])
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# ============================================================
# 10. SIX-PANEL ACADEMIC FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 10),
    constrained_layout=True,
)

fig.suptitle(
    "Audio Signal Processing Pipeline",
    fontsize=16,
    fontweight="bold",
    color=BLUE["navy"],
)

# ------------------------------------------------------------
# (a) Original waveform
# ------------------------------------------------------------

ax = axes[0, 0]

ax.plot(
    time_seconds,
    audio,
    color=BLUE["navy"],
    linewidth=0.65,
)

ax.set_title("(a) Original Audio Waveform", loc="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, duration)
format_axis(ax)


# ------------------------------------------------------------
# (b) Pre-emphasis output
# ------------------------------------------------------------

ax = axes[0, 1]

ax.plot(
    time_seconds,
    processed_audio,
    color=BLUE["medium"],
    linewidth=0.65,
)

ax.set_title(
    "(b) Pre-emphasis Output "
    f"({'Enabled' if PRE_EMPHASIS_ENABLED else 'Disabled'})",
    loc="left",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, duration)
format_axis(ax)


# ------------------------------------------------------------
# (c) Overlapping frames
# ------------------------------------------------------------

ax = axes[1, 0]

display_duration = min(duration, 0.10)
display_samples = int(display_duration * SAMPLE_RATE)

ax.plot(
    time_seconds[:display_samples],
    processed_audio[:display_samples],
    color=BLUE["navy"],
    linewidth=0.8,
    zorder=3,
)

frame_shades = [
    BLUE["pale"],
    BLUE["light"],
    BLUE["bright"],
]

for index in range(min(3, number_of_frames)):
    start = index * HOP_LENGTH
    end = start + FRAME_LENGTH

    ax.axvspan(
        start / SAMPLE_RATE,
        end / SAMPLE_RATE,
        color=frame_shades[index],
        alpha=0.55,
        linewidth=0,
        label=f"Frame {index + 1}",
        zorder=1,
    )

ax.set_title("(c) Overlapping Frame Segmentation", loc="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, display_duration)
ax.legend(frameon=False, ncol=3, loc="upper right")
format_axis(ax)


# ------------------------------------------------------------
# (d) Hamming window coefficients
# ------------------------------------------------------------

ax = axes[1, 1]

ax.plot(
    frame_time_ms,
    window,
    color=BLUE["bright"],
    linewidth=2.0,
)

ax.fill_between(
    frame_time_ms,
    window,
    color=BLUE["pale"],
    alpha=0.65,
)

ax.set_title("(d) Hamming Window", loc="left")
ax.set_xlabel("Time within Frame (ms)")
ax.set_ylabel("Window Coefficient")
ax.set_xlim(0, FRAME_DURATION_MS)
ax.set_ylim(0, 1.05)
format_axis(ax)


# ------------------------------------------------------------
# (e) Windowed audio frame
# ------------------------------------------------------------

ax = axes[2, 0]

ax.plot(
    frame_time_ms,
    selected_frame,
    color=BLUE["light"],
    linewidth=1.0,
    label="Before windowing",
)

ax.plot(
    frame_time_ms,
    windowed_frame,
    color=BLUE["navy"],
    linewidth=1.4,
    label="After windowing",
)

ax.set_title("(e) Windowed Audio Frame", loc="left")
ax.set_xlabel("Time within Frame (ms)")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, FRAME_DURATION_MS)
ax.legend(frameon=False)
format_axis(ax)


# ------------------------------------------------------------
# (f) Power spectrum
# ------------------------------------------------------------

ax = axes[2, 1]

ax.plot(
    frequencies,
    power_spectrum,
    color=BLUE["dark"],
    linewidth=1.2,
)

ax.fill_between(
    frequencies,
    power_spectrum,
    color=BLUE["light"],
    alpha=0.35,
)

ax.set_title("(f) Power Spectrum", loc="left")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_xlim(0, SAMPLE_RATE / 2)

# A logarithmic scale can help reveal weaker spectral components.
# Use linear scale instead if required for a specific comparison.
if np.any(power_spectrum > 0):
    ax.set_yscale("symlog", linthresh=1e-10)

format_axis(ax)


# ============================================================
# 11. EXPORT FIGURE
# ============================================================

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

print("=" * 60)
print("SIGNAL PROCESSING SUMMARY")
print("=" * 60)
print(f"Input file:       {AUDIO_PATH.name}")
print(f"Sample rate:      {SAMPLE_RATE} Hz")
print(f"Duration:         {duration:.3f} s")
print(f"Frame length:     {FRAME_LENGTH} samples")
print(f"Frame overlap:    {FRAME_OVERLAP} samples")
print(f"Hop length:       {HOP_LENGTH} samples")
print(f"Number of frames: {number_of_frames}")
print(f"FFT size:         {FFT_SIZE}")
print(f"FFT bins:         {len(power_spectrum)}")
print(f"Pre-emphasis:     {PRE_EMPHASIS_ENABLED}")
print(f"\nPNG saved to:\n{OUTPUT_PNG}")
print(f"\nPDF saved to:\n{OUTPUT_PDF}")

plt.show()
plt.close(fig)
