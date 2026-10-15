"""The episode: one pushing run in the shared format that every data source produces.

Everything after the dataset adapters sees only this format. Pusher actions are not
stored: they depend on the step, so windowing derives them from the pusher positions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "held-out")
KINDS = ("reference", "push", "near-miss")


@dataclass
class Episode:
    name: str
    frames: np.ndarray  # (T, H, W, 3) uint8, top-down view
    timestamps: np.ndarray  # (T,) seconds, evenly spaced
    pusher_pos: np.ndarray  # (T, 2) pusher position, table units
    object_pos: np.ndarray  # (T, 2) object position, table units
    workspace: float  # width of the square workspace, table units
    unit: str  # name of the table unit, e.g. "cm" or "Push-T units"
    split: str  # one of SPLITS
    kind: str  # one of KINDS

    def __post_init__(self):
        n = len(self.frames)
        assert self.frames.dtype == np.uint8 and self.frames.ndim == 4, self.name
        assert self.timestamps.shape == (n,), self.name
        assert self.pusher_pos.shape == (n, 2), self.name
        assert self.object_pos.shape == (n, 2), self.name
        assert self.split in SPLITS, self.split
        assert self.kind in KINDS, self.kind


def save_episodes(episodes: list[Episode], folder: Path) -> None:
    """Write one .npz file per episode into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        meta = dict(workspace=ep.workspace, unit=ep.unit, split=ep.split, kind=ep.kind)
        np.savez(
            folder / f"{ep.name}.npz",
            frames=ep.frames,
            timestamps=ep.timestamps,
            pusher_pos=ep.pusher_pos,
            object_pos=ep.object_pos,
            meta=json.dumps(meta),
        )


def load_episodes(folder: Path) -> list[Episode]:
    """Read every episode in `folder`, sorted by name. This order is the canonical one."""
    episodes = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            episodes.append(
                Episode(
                    name=path.stem,
                    frames=f["frames"],
                    timestamps=f["timestamps"],
                    pusher_pos=f["pusher_pos"],
                    object_pos=f["object_pos"],
                    **meta,
                )
            )
    return episodes
