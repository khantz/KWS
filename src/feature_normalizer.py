import numpy as np


class FeatureNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features):

        if features.ndim != 3:
            raise ValueError(
                "Expected features with shape "
                "(samples, frames, coefficients)."
            )

        # Calculate statistics independently for
        # each MFCC coefficient across all frames/samples.
        flattened = features.reshape(
            -1,
            features.shape[-1]
        )

        self.mean = np.mean(
            flattened,
            axis=0,
            keepdims=True
        )

        self.std = np.std(
            flattened,
            axis=0,
            keepdims=True
        )

        self.std = np.maximum(
            self.std,
            1e-8
        )

    def transform(self, features):

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Normalizer must be fitted before transform()."
            )

        return (
            (
                features - self.mean
            ) / self.std
        ).astype(np.float32)

    def fit_transform(self, features):

        self.fit(features)

        return self.transform(features)

    def save(self, output_path):

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        np.savez(
            output_path,
            mean=self.mean,
            std=self.std
        )