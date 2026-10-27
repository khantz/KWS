import os
import random
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

FEATURE_ROOT = r"C:\Users\M S I\PycharmProjects\KWS\feature\mfcc_feature"

SPLIT = "train"

CLASS_NAMES = [
    "no",
    "off",
    "on",
    "one",
    "stop",
    "three",
    "two",
    "unknown",
    "yes"
]

EXPECTED_FEATURE_SHAPE = (71, 13)

# Number of samples used from each class for visualization
SAMPLES_PER_CLASS = 500

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Load MFCC features
# ============================================================

def load_features():

    all_features = []
    all_labels = []

    invalid_files = []

    for class_name in CLASS_NAMES:

        class_directory = os.path.join(
            FEATURE_ROOT,
            SPLIT,
            class_name
        )

        if not os.path.isdir(class_directory):
            print(f"[WARNING] Directory not found: {class_directory}")
            continue

        feature_files = [
            filename
            for filename in os.listdir(class_directory)
            if filename.lower().endswith(".npy")
        ]

        valid_features = []

        for filename in feature_files:

            file_path = os.path.join(
                class_directory,
                filename
            )

            try:
                feature = np.load(file_path)

                # --------------------------------------------
                # Validate feature shape
                # --------------------------------------------

                if feature.shape != EXPECTED_FEATURE_SHAPE:
                    invalid_files.append(
                        (file_path, feature.shape)
                    )
                    continue

                # --------------------------------------------
                # Validate numerical values
                # --------------------------------------------

                if not np.isfinite(feature).all():
                    invalid_files.append(
                        (file_path, "NaN/Inf")
                    )
                    continue

                valid_features.append(
                    feature.astype(np.float32)
                )

            except Exception as error:

                invalid_files.append(
                    (file_path, str(error))
                )

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        if len(valid_features) > SAMPLES_PER_CLASS:

            selected_features = random.sample(
                valid_features,
                SAMPLES_PER_CLASS
            )

        else:

            selected_features = valid_features

        print(
            f"{class_name:8s} | "
            f"valid: {len(valid_features):5d} | "
            f"selected: {len(selected_features):4d}"
        )

        # ----------------------------------------------------
        # Store features and labels
        # ----------------------------------------------------

        for feature in selected_features:

            all_features.append(
                feature.reshape(-1)
            )

            all_labels.append(class_name)

    return (
        np.asarray(all_features, dtype=np.float32),
        np.asarray(all_labels)
    ), invalid_files


# ============================================================
# Load dataset
# ============================================================

(X, labels), invalid_files = load_features()


# ============================================================
# Dataset validation
# ============================================================

print("\n============================================================")
print("Dataset Summary")
print("============================================================")

print(f"Feature matrix shape : {X.shape}")
print(f"Number of samples    : {len(X)}")
print(f"Feature dimensions    : {X.shape[1]}")
print(f"Expected dimensions   : {71 * 13}")

if X.shape[1] != 71 * 13:
    raise ValueError(
        f"Unexpected feature dimension: {X.shape[1]}"
    )

if len(X) == 0:
    raise ValueError(
        "No valid MFCC features were loaded."
    )

print(
    f"Invalid files        : {len(invalid_files)}"
)


# ============================================================
# Standardization
# ============================================================
#
# Each flattened MFCC vector contains:
#
# 71 frames × 13 coefficients = 923 features
#
# StandardScaler transforms each feature dimension so that
# it has approximately zero mean and unit variance.
# ============================================================

print("\nApplying feature normalization...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = X_scaled.astype(np.float32)

print("Normalization completed.")


# ============================================================
# UMAP dimensionality reduction
# ============================================================

print("\nRunning UMAP...")

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.05,
    metric="euclidean",
    random_state=RANDOM_SEED
)

X_umap = reducer.fit_transform(X_scaled)

print("UMAP completed.")


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(11, 8))

# Use a qualitative colormap with 9 distinct colors
cmap = plt.get_cmap("tab10")

for class_index, class_name in enumerate(CLASS_NAMES):

    class_mask = labels == class_name

    if not np.any(class_mask):
        continue

    plt.scatter(
        X_umap[class_mask, 0],
        X_umap[class_mask, 1],

        s=18,
        alpha=0.65,

        color=cmap(class_index),

        label=class_name
    )


# ============================================================
# Figure formatting
# ============================================================

plt.title(
    "UMAP Visualization of MFCC Feature Space",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel(
    "UMAP Dimension 1",
    fontsize=12
)

plt.ylabel(
    "UMAP Dimension 2",
    fontsize=12
)

plt.legend(
    title="Keyword Class",
    fontsize=10,
    title_fontsize=10,

    loc="best",
    frameon=True
)

plt.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25
)

plt.tight_layout()


# ============================================================
# Save figure
# ============================================================

OUTPUT_FILE = os.path.join(
    FEATURE_ROOT,
    "umap_mfcc_feature_distribution.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print("\n============================================================")
print("UMAP Visualization Completed")
print("============================================================")
print(f"Saved to: {OUTPUT_FILE}")

plt.show()