from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# Configuration
# ============================================================

FEATURE_ROOT = Path(
    r"C:/Users/M S I/PycharmProjects/KWS/feature/mfcc_feature"
)

SPLITS = [
    "train",
    "val",
    "test"
]

CLASS_NAMES = [
    "no",
    "off",
    "on",
    "one",
    "stop",
    "three",
    "two",
    "unknown",
    "yes",
]

EXPECTED_SHAPE = (71, 13)


# ============================================================
# Load features
# ============================================================

def load_features():

    features = []
    labels = []

    print("\nFeature root:")
    print(FEATURE_ROOT)

    if not FEATURE_ROOT.exists():
        raise FileNotFoundError(
            f"Feature directory does not exist:\n"
            f"{FEATURE_ROOT}"
        )

    print("\n" + "=" * 70)
    print("SEARCHING FOR MFCC FEATURES")
    print("=" * 70)

    for split in SPLITS:

        split_directory = FEATURE_ROOT / split

        print(f"\n[{split.upper()}]")

        if not split_directory.exists():
            print(
                f"  [WARNING] Missing directory: "
                f"{split_directory}"
            )
            continue

        for class_name in CLASS_NAMES:

            class_directory = (
                split_directory / class_name
            )

            if not class_directory.exists():
                print(
                    f"  [WARNING] Missing class directory: "
                    f"{class_directory}"
                )
                continue

            # rglob searches recursively
            feature_files = sorted(
                class_directory.rglob("*.npy")
            )

            print(
                f"  {class_name:8s}: "
                f"{len(feature_files):6d} files"
            )

            for feature_file in feature_files:

                try:

                    feature = np.load(
                        feature_file
                    )

                    if feature.shape != EXPECTED_SHAPE:
                        print(
                            f"[WARNING] Skipping "
                            f"{feature_file}"
                        )
                        print(
                            f"           Shape: "
                            f"{feature.shape}"
                        )
                        continue

                    # 70 × 13 → 910
                    feature_vector = (
                        feature.reshape(-1)
                    )

                    features.append(
                        feature_vector
                    )

                    labels.append(
                        class_name
                    )

                except Exception as error:

                    print(
                        f"[ERROR] Could not load: "
                        f"{feature_file}"
                    )

                    print(
                        f"        {error}"
                    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print("\n" + "=" * 70)
    print("LOADING SUMMARY")
    print("=" * 70)

    print(
        f"Total features : {len(features)}"
    )

    print(
        f"Feature matrix : {features.shape}"
    )

    print(
        f"Labels         : {len(labels)}"
    )

    if len(features) == 0:

        raise RuntimeError(
            "\nNo MFCC .npy files were found.\n\n"
            "Check FEATURE_ROOT and your directory structure."
        )

    return features, labels


# ============================================================
# PCA
# ============================================================

def reduce_to_two_dimensions(features):

    print("\nPerforming PCA...")

    pca = PCA(
        n_components=2,
        random_state=42
    )

    reduced_features = pca.fit_transform(
        features
    )

    explained = (
        pca.explained_variance_ratio_
    )

    print(
        f"PC1 explained variance: "
        f"{explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{explained[1] * 100:.2f}%"
    )

    print(
        f"Total explained variance: "
        f"{explained.sum() * 100:.2f}%"
    )

    return reduced_features


# ============================================================
# Scatter plot
# ============================================================

def plot_features(
    reduced_features,
    labels
):

    print("\nCreating scatter plot...")

    plt.figure(
        figsize=(12, 8)
    )

    # Different color automatically assigned
    # to every class
    for class_name in CLASS_NAMES:

        indices = [
            i
            for i, label in enumerate(labels)
            if label == class_name
        ]

        if not indices:
            continue

        points = reduced_features[
            indices
        ]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            label=class_name,
            alpha=0.45,
            s=10
        )

    plt.title(
        "MFCC Feature Distribution by Class"
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(
        True,
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MFCC FEATURE DISTRIBUTION")
    print("=" * 70)

    features, labels = (
        load_features()
    )

    reduced_features = (
        reduce_to_two_dimensions(
            features
        )
    )

    plot_features(
        reduced_features,
        labels
    )


if __name__ == "__main__":
    main()