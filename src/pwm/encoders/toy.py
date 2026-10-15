"""Toy encoder: a cheap stand-in for a frozen image encoder, used in tests.

The frame embedding is the mean colour of each patch on an 8 x 8 grid, so it keeps the
patch-grid shape of the real encoders: (T, patches, dims) = (T, 64, 3).
"""

import numpy as np

GRID = 8


class ToyEncoder:
    name = "toy"

    def encode(self, frames: np.ndarray) -> np.ndarray:
        t, h, w, c = frames.shape
        patches = frames.reshape(t, GRID, h // GRID, GRID, w // GRID, c)
        return (patches.mean(axis=(2, 4)) / 255).reshape(t, GRID * GRID, c).astype(np.float32)
