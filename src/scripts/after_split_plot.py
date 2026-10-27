from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\M S I\PycharmProjects\KWS")

SPLIT_DIR = ROOT / "dataset" / "after_splited"
PREP_DIR  = ROOT / "dataset" / "after_preprocessed"

OUTPUT_DIR = ROOT / "results" / "dataset_comparison" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# ------------------------------------------------------------
# Blue palette
# ------------------------------------------------------------
C_RAW  = "#9ECAE1"   # light blue -> after_splited
C_PREP = "#08519C"   # dark blue  -> after_preprocessed
C_LINE = "#08306B"
C_TEXT = "#08306B"
C_GRID = "#D9E6F2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})


def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {name}.png / .pdf")


# ============================================================
# COUNTING
# ============================================================

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS


def count_files(base: Path) -> dict:
    if not base.is_dir():
        raise FileNotFoundError(f"Folder not found: {base}")

    subfolders = [p for p in base.iterdir() if p.is_dir()]
    counts = {}

    if subfolders:
        for folder in sorted(subfolders, key=lambda p: p.name.lower()):
            n = sum(1 for f in folder.rglob("*")
                    if f.is_file() and is_audio(f))
            counts[folder.name] = n
    else:
        n = sum(1 for f in base.rglob("*")
                if f.is_file() and is_audio(f))
        counts["all"] = n

    return counts


def normalize_split_name(name: str) -> str:
    n = name.strip().lower()
    if n in {"val", "validation", "valid"}:
        return "validation"
    if n in {"train", "training"}:
        return "train"
    if n in {"test", "testing"}:
        return "test"
    return n


def normalize_counts(counts: dict) -> dict:
    out = defaultdict(int)
    for k, v in counts.items():
        out[normalize_split_name(k)] += v
    return dict(out)


def order_splits(splits):
    return sorted(
        splits,
        key=lambda s: (s != "train", s != "validation", s != "test", s),
    )


# ============================================================
# FIGURE — grouped bars: raw vs preprocessed per split
# ============================================================

def figure_comparison(raw: dict, prep: dict):
    splits = order_splits(set(raw) | set(prep))

    raw_vals  = [raw.get(s, 0) for s in splits]
    prep_vals = [prep.get(s, 0) for s in splits]

    x = np.arange(len(splits))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(x - w/2, raw_vals, w,
                   color=C_RAW, edgecolor=C_LINE,
                   linewidth=1.0, label="After splitting")
    bars2 = ax.bar(x + w/2, prep_vals, w,
                   color=C_PREP, edgecolor=C_LINE,
                   linewidth=1.0, label="After preprocessing")

    top = max(raw_vals + prep_vals) or 1

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + top * 0.012,
                f"{int(h):,}",
                ha="center", va="bottom",
                fontsize=9, color=C_TEXT, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of audio files")
    ax.set_title("Dataset File Counts: After Splitting vs After Preprocessing",
                 loc="left", color=C_TEXT, fontweight="bold", pad=12)

    ax.set_ylim(0, top * 1.20)
    ax.yaxis.grid(True, linestyle=":", color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Small note confirming identical counts.
    total_raw  = sum(raw_vals)
    total_prep = sum(prep_vals)
    if total_raw == total_prep:
        ax.text(
            0.99, 0.97,
            "Counts identical — no files lost",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, color=C_TEXT, style="italic",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#E7F0FA",
                      edgecolor=C_LINE, linewidth=0.8),
        )

    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    save_fig(fig, "Figure_Dataset_Counts")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    raw_counts  = normalize_counts(count_files(SPLIT_DIR))
    prep_counts = normalize_counts(count_files(PREP_DIR))

    print("\nAfter splitting (raw):")
    for k, v in raw_counts.items():
        print(f"  {k:<12} {v:>8,}")

    print("\nAfter preprocessing:")
    for k, v in prep_counts.items():
        print(f"  {k:<12} {v:>8,}")

    figure_comparison(raw_counts, prep_counts)

    print(f"\nFigure saved to:\n{OUTPUT_DIR}")