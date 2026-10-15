"""Probe: a linear read-out of the object position from a frame embedding (ridge regression)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Probe:
    weights: np.ndarray  # (features, 2)
    bias: np.ndarray  # (2,)

    @classmethod
    def fit(cls, embeddings: np.ndarray, positions: np.ndarray, alpha: float = 1e-3) -> "Probe":
        """Fit on real embeddings (N, ...) and object positions (N, 2).

        `alpha` is relative to the mean feature variance, so it means the same for any encoder.
        """
        x = embeddings.reshape(len(embeddings), -1).astype(np.float64)
        x_mean, y_mean = x.mean(0), positions.mean(0)
        x, y = x - x_mean, positions - y_mean
        n, f = x.shape
        ridge = alpha * (x**2).sum() / f
        if f <= n:
            weights = np.linalg.solve(x.T @ x + ridge * np.eye(f), x.T @ y)
        else:  # fewer frames than features: solve the smaller, equivalent system
            weights = x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(n), y)
        return cls(weights=weights, bias=y_mean - x_mean @ weights)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Object positions (N, 2) read from embeddings (N, ...)."""
        x = embeddings.reshape(len(embeddings), -1).astype(np.float64)
        return x @ self.weights + self.bias
