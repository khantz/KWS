
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\M S I\PycharmProjects\KWS")

RAW_DIR = ROOT / "dataset" / "raw"
SPLIT_DIR = ROOT / "dataset" / "after_splited"

OUTPUT_DIR = ROOT / "result_png" / "dataset_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav"}


def audio_files(directory):
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def find_class(file_path, root):
    """
    Use the immediate parent folder as the class label.
    For split data, this is usually:
    train/no/file.wav -> no
    """
    relative = file_path.relative_to(root)
    parts = relative.parts

    if len(parts) >= 2 and parts[0].lower() in {
        "train", "training", "val", "valid",
        "validation", "test", "testing"
    }:
        return parts[1]

    return file_path.parent.name


def count_classes(files, root):
    counts = Counter(find_class(p, root) for p in files)
    return counts


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# --------------------------------------------------
# 1. Count raw dataset files by class
# --------------------------------------------------
raw_files = audio_files(RAW_DIR)
raw_counts = count_classes(raw_files, RAW_DIR)

print("Raw WAV files:", len(raw_files))
print("Raw class counts:")
print(pd.Series(raw_counts).sort_index())

raw_series = pd.Series(raw_counts).sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
raw_series.sort_values().plot(
    kind="barh",
    ax=ax,
)
ax.set_title("Raw Dataset Class Distribution")
ax.set_xlabel("Number of WAV files")
ax.set_ylabel("Class")
ax.grid(axis="x", alpha=0.3)
save_plot("figure4_raw_class_distribution.png")


# --------------------------------------------------
# 2. Count train / validation / test files
# --------------------------------------------------
split_names = {
    "train": ["train", "training"],
    "validation": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}

split_dirs = {}

for split_name, aliases in split_names.items():
    matches = [
        p for p in SPLIT_DIR.iterdir()
        if p.is_dir() and p.name.lower() in aliases
    ]
    if matches:
        split_dirs[split_name] = matches[0]

if not split_dirs:
    raise FileNotFoundError(
        f"No train/validation/test directories found in {SPLIT_DIR}. "
        "Check the actual folder names."
    )

split_totals = {}
split_class_counts = {}

for split_name, directory in split_dirs.items():
    files = audio_files(directory)
    counts = count_classes(files, directory)

    split_totals[split_name] = len(files)
    split_class_counts[split_name] = counts

print("\nSplit totals:")
print(pd.Series(split_totals))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    list(split_totals.keys()),
    list(split_totals.values()),
)
ax.bar_label(bars, fmt="%d", padding=3)
ax.set_title("Dataset Distribution After Splitting")
ax.set_ylabel("Number of WAV files")
ax.grid(axis="y", alpha=0.3)
save_plot("figure5_split_distribution.png")


# Plot split counts per class
split_df = pd.DataFrame(split_class_counts).fillna(0).astype(int)
split_df = split_df.sort_index()

ax = split_df.plot(
    kind="bar",
    figsize=(12, 6),
)
ax.set_title("Class Distribution Across Dataset Splits")
ax.set_xlabel("Class")
ax.set_ylabel("Number of WAV files")
ax.legend(title="Subset")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "figure5b_class_distribution_by_split.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# 3. Compare original vs augmented training files
# --------------------------------------------------
train_dir = split_dirs.get("train")

if train_dir is None:
    print(
        "\nNo training directory detected. "
        "Skipping augmentation plot."
    )
else:
    train_files = audio_files(train_dir)

    original_counts = Counter()
    augmented_counts = Counter()

    for path in train_files:
        class_name = find_class(path, train_dir)

        if "_aug_" in path.stem.lower():
            augmented_counts[class_name] += 1
        else:
            original_counts[class_name] += 1

    all_classes = sorted(
        set(original_counts) | set(augmented_counts)
    )

    augmentation_df = pd.DataFrame(
        {
            "Original training files": [
                original_counts[c] for c in all_classes
            ],
            "Augmented files": [
                augmented_counts[c] for c in all_classes
            ],
        },
        index=all_classes,
    )

    print("\nTraining-set original vs augmented:")
    print(augmentation_df)
    print(
        "Original training files:",
        sum(original_counts.values()),
    )
    print(
        "Generated augmented files:",
        sum(augmented_counts.values()),
    )
    print(
        "Total training files:",
        len(train_files),
    )

    ax = augmentation_df.plot(
        kind="bar",
        figsize=(12, 6),
    )
    ax.set_title(
        "Original and Augmented Samples in the Training Set"
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of WAV files")
    ax.legend(title="Sample type")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figure6_augmentation_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

print("\nPlots saved to:", OUTPUT_DIR)
