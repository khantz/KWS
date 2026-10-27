
from pathlib import Path

import numpy as np
import yaml

from src.feature_loader import MFCCFeatureLoader
from src.feature_normalizer import FeatureNormalizer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\M S I\PycharmProjects\KWS"
)

MODELS_DIR = PROJECT_ROOT / "Models"

CONFIGS = {
    "CNN1D": PROJECT_ROOT / "configs" / "oned_cnn_training_config.yaml",
    "DSCNN": PROJECT_ROOT / "configs" / "dscnn_training_config.yaml",
}

OUTPUTS = {
    "CNN1D": MODELS_DIR / "CNN1D_normalization.npz",
    "DSCNN": MODELS_DIR / "DSCNN_normalization.npz",
}


# ============================================================
# Recreate normalization parameters
# ============================================================

def recreate_normalization(model_name, config_path, output_path):
    print("\n" + "=" * 60)
    print(f"Recreating normalization: {model_name}")
    print("=" * 60)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    # Load the original training configuration.
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Use the same feature loader as the original training script.
    loader = MFCCFeatureLoader(
        feature_directory=config["dataset"]["feature_directory"],
        class_names=config["classes"],
    )

    # Load ONLY the training split.
    x_train, y_train_text = loader.load_split(
        config["dataset"]["training"]["directory"]
    )

    x_train = np.asarray(x_train, dtype=np.float32)

    if x_train.size == 0:
        raise ValueError(
            f"{model_name}: training features are empty."
        )

    print(f"Training features: {x_train.shape}")
    print(f"Training samples:  {len(x_train)}")
    print(f"Feature dtype:     {x_train.dtype}")

    # Fit the same normalizer used during original training.
    normalizer = FeatureNormalizer()
    _ = normalizer.fit_transform(x_train)

    # Save using the original FeatureNormalizer implementation.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"{model_name}: normalization file was not saved correctly."
        )

    print(f"Saved: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")

    # Verify that the saved file is a readable NumPy archive.
    with np.load(output_path, allow_pickle=False) as archive:
        print(f"Archive keys: {archive.files}")

        for key in archive.files:
            values = archive[key]
            print(
                f"  {key}: shape={values.shape}, "
                f"dtype={values.dtype}"
            )

    print(f"{model_name}: normalization file verified.")


def main():
    for model_name in ("CNN1D", "DSCNN"):
        recreate_normalization(
            model_name=model_name,
            config_path=CONFIGS[model_name],
            output_path=OUTPUTS[model_name],
        )

    print("\nBoth normalization files have been recreated.")


if __name__ == "__main__":
    main()

