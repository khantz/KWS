import numpy as np


class MFCCFeatureValidator:

    def __init__(self, expected_shape=(71, 13)):
        self.expected_shape = tuple(expected_shape)

    def validate(self, features, split_name="dataset"):

        if not isinstance(features, np.ndarray):
            raise TypeError(
                f"{split_name}: features must be a NumPy array."
            )

        if features.ndim != 3:
            raise ValueError(
                f"{split_name}: expected 3 dimensions "
                f"(samples, frames, coefficients), "
                f"got {features.ndim}."
            )

        actual_shape = features.shape[1:]

        if actual_shape != self.expected_shape:
            raise ValueError(
                f"{split_name}: expected feature shape "
                f"{self.expected_shape}, "
                f"got {actual_shape}."
            )

        if len(features) == 0:
            raise ValueError(
                f"{split_name}: feature dataset is empty."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                f"{split_name}: NaN or Inf detected."
            )

        print(
            f"[VALID] {split_name}: "
            f"{features.shape[0]} samples, "
            f"feature shape {actual_shape}"
        )

        return True