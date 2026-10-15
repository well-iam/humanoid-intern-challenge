"""Probe: a linear read-out of the object position from a frame embedding (ridge regression).

A frame embedding of the real encoders has up to 16 x 16 x 1024 = 262,144 features, more
than there are training frames, so the probe is solved in its dual form: through the
frames-by-frames Gram matrix, built one block of features at a time. The embeddings are
never all in memory at once, so they can stay memory-mapped on disk.

Each feature is standardised first, so a few channels with huge activations (common in
ViTs) neither swamp the float32 Gram matrix nor decide the ridge penalty alone.

The ridge strength is chosen on a few training episodes held back for validation, then the
probe is refitted on all training episodes. Held-out episodes are never seen.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# Ridge strengths tried, relative to the mean squared norm of a standardised frame embedding.
RIDGE_GRID = [10.0**k for k in range(-8, 2)]
VALIDATION_EVERY = 5  # every 5th training episode helps choose the ridge strength
BLOCK_BYTES = 2**30  # size of one block of features in float32
GPU_ABOVE = 2**26  # use the GPU (if any) when frames x features is above this


@dataclass
class Probe:
    weights: np.ndarray  # (features, 2) float32
    bias: np.ndarray  # (2,)
    ridge: float  # chosen ridge strength, relative to the mean squared norm of an embedding

    @classmethod
    def fit(cls, embeddings: Sequence[np.ndarray], positions: Sequence[np.ndarray]) -> "Probe":
        """Fit on the real embeddings (T, ...) and object positions (T, 2) of each training episode."""
        views = [e.reshape(len(e), -1) for e in embeddings]
        n, f = sum(len(v) for v in views), views[0].shape[1]
        device = "cuda" if torch.cuda.is_available() and n * f > GPU_ABOVE else "cpu"
        y = torch.from_numpy(np.concatenate(positions)).double().to(device)
        blocks = _feature_blocks(n, f)

        # Pass 1: feature means and spreads, and the Gram matrix of the standardised embeddings.
        mean, std = np.zeros(f, np.float32), np.ones(f, np.float32)
        gram = torch.zeros(n, n, dtype=torch.float64, device=device)
        for cols in blocks:
            x = _block(views, cols, device)
            m, s = x.mean(0), x.std(0)
            s[s == 0] = 1
            mean[cols], std[cols] = m.cpu().numpy(), s.cpu().numpy()
            x = (x - m) / s
            gram += (x @ x.T).double()

        ridge = _choose_ridge(gram, y, [len(v) for v in views])

        # Pass 2: the primal weights from the dual solution a. On standardised features they
        # are X^T a; dividing by the spread once more makes them apply to the raw embedding.
        y_mean = y.mean(0)
        eye = torch.eye(n, dtype=gram.dtype, device=device)
        a = torch.linalg.solve(gram + ridge * _scale(gram) * eye, y - y_mean).float()
        weights = np.zeros((f, 2), np.float32)
        for cols in blocks:
            x = _block(views, cols, device) - torch.from_numpy(mean[cols]).to(device)
            weights[cols] = (x.T @ a).cpu().numpy() / std[cols, None] ** 2
        bias = y_mean.cpu().numpy() - mean.astype(np.float64) @ weights
        return cls(weights=weights, bias=bias, ridge=ridge)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Object positions (N, 2) read from embeddings (N, ...)."""
        x = embeddings.reshape(len(embeddings), -1)
        rows = max(1, BLOCK_BYTES // (4 * x.shape[1]))
        out = [x[i : i + rows].astype(np.float32) @ self.weights for i in range(0, len(x), rows)]
        return np.concatenate(out).astype(np.float64) + self.bias if out else np.zeros((0, 2))


def _feature_blocks(n: int, f: int) -> list[slice]:
    width = max(1, BLOCK_BYTES // (4 * n))
    return [slice(c, min(c + width, f)) for c in range(0, f, width)]


def _block(views: list[np.ndarray], cols: slice, device: str) -> torch.Tensor:
    x = np.concatenate([v[:, cols] for v in views]).astype(np.float32)
    return torch.from_numpy(x).to(device)


def _scale(gram: torch.Tensor) -> float:
    """Mean squared norm of a standardised embedding, so ridge strengths mean the same for any encoder."""
    return float(gram.diagonal().mean())


def _choose_ridge(gram: torch.Tensor, y: torch.Tensor, lengths: list[int]) -> float:
    """The ridge strength with the lowest mean position error on the validation episodes."""
    episode = np.repeat(np.arange(len(lengths)), lengths)
    is_val = torch.from_numpy(episode % VALIDATION_EVERY == VALIDATION_EVERY - 1).to(gram.device)
    if not is_val.any() or is_val.all():
        raise ValueError(f"the probe needs at least {VALIDATION_EVERY} training episodes")
    fit, val = ~is_val, is_val

    # Re-centre the Gram matrix on the fitting frames only (centring is a shift of the features).
    g_fit = gram[fit][:, fit]
    row_mean = gram[:, fit].mean(1)  # each frame's mean product with the fitting frames
    total = g_fit.mean()
    k_fit = g_fit - row_mean[fit, None] - row_mean[None, fit] + total
    k_val = gram[val][:, fit] - row_mean[val, None] - row_mean[None, fit] + total
    y_mean = y[fit].mean(0)

    values, vectors = torch.linalg.eigh(k_fit)
    projected = vectors.T @ (y[fit] - y_mean)
    scale = _scale(k_fit)
    errors = []
    for ridge in RIDGE_GRID:
        a = vectors @ (projected / (values.clamp(min=0) + ridge * scale)[:, None])
        errors.append(float(torch.linalg.norm(k_val @ a + y_mean - y[val], dim=1).mean()))
    return RIDGE_GRID[int(np.argmin(errors))]
