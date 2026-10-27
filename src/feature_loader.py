from pathlib import Path
import numpy as np


class MFCCFeatureLoader:
    def __init__(self, feature_directory, class_names):
        self.feature_directory = Path(feature_directory)
        self.class_names = class_names

    def load_split(self, split_directory):
        split_path = self.feature_directory / split_directory

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Feature split directory not found: {split_path}"
            )

        features = []
        labels = []

        print(f"\nLoading split: {split_directory}")

        for class_name in self.class_names:
            class_path = split_path / class_name

            if not class_path.is_dir():
                raise FileNotFoundError(
                    f"Class directory not found: {class_path}"
                )

            feature_files = sorted(class_path.glob("*.npy"))

            print(
                f"  {class_name:<8} : "
                f"{len(feature_files):>5} files"
            )

            for feature_file in feature_files:
                feature = np.load(feature_file)

                features.append(feature.astype(np.float32))
                labels.append(class_name)

        if not features:
            raise ValueError(
                f"No feature files found in: {split_path}"
            )

        print(f"  Finished {split_directory}: {len(features)} samples")

        return (
            np.asarray(features, dtype=np.float32),
            np.asarray(labels)
        )

    def load_train(self):
        return self.load_split("train")

    def load_validation(self):
        return self.load_split("val")

    def load_test(self):
        return self.load_split("test")