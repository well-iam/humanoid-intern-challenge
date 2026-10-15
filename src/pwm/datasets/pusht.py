"""Push-T adapter: the Diffusion Policy replay file -> episodes.

The file is downloaded once from its original URL into the raw folder and read in place,
straight from the zip, with zarr 3 (not Diffusion Policy's own loader). It holds 206
human demonstrations at 10 Hz, seen from above in a 512 x 512 window.

- Object position: the centre of the T. The stored block x, y is the midpoint of the
  crossbar's outer edge, so the centre is 45 units further along the T's stem.
- Pusher position: the agent position. Pusher actions are not the recorded `action` (a
  target for the controller); windowing derives them as the observed displacement.
- Frames: the stored 96 x 96 images are float32 with values 0-255; they are cast to uint8
  one episode at a time.
"""

import urllib.request
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import ZipStore

from pwm.episodes import Episode

URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_PATH = "pusht/pusht_cchi_v7_replay.zarr"
WORKSPACE = 512.0
UNIT = "Push-T units"
FPS = 10
T_CENTRE_OFFSET = 45.0  # from the crossbar's outer edge to the centre of the T

STEP = 0.5  # seconds: every 5th frame, so the object visibly moves between steps
MOVE_THRESHOLD = 5.0  # Push-T units
N_TRAIN, N_HELD_OUT, SEED = 100, 50, 0


def build(raw_dir: Path) -> list[Episode]:
    """Dataset-adapter entry point: 100 training and 50 held-out episodes, chosen by a fixed seed."""
    group = zarr.open_group(ZipStore(download(raw_dir), mode="r"), mode="r", path=ZARR_PATH)
    ends = group["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])

    chosen = np.random.default_rng(SEED).permutation(len(ends))[: N_TRAIN + N_HELD_OUT]
    held_out = set(chosen[N_TRAIN:].tolist())

    episodes = []
    for index in sorted(chosen.tolist()):
        rows = slice(starts[index], ends[index])
        state = group["data/state"][rows]
        episodes.append(
            Episode(
                name=f"pusht-{index:03d}",
                frames=np.round(group["data/img"][rows]).clip(0, 255).astype(np.uint8),
                timestamps=np.arange(len(state)) / FPS,
                pusher_pos=state[:, 0:2].astype(np.float64),
                object_pos=t_centre(state[:, 2:5]),
                workspace=WORKSPACE,
                unit=UNIT,
                split="held-out" if index in held_out else "train",
                kind="reference",
            )
        )
    return episodes


def t_centre(block: np.ndarray) -> np.ndarray:
    """Centre of the T from the stored block pose (x, y, angle): (x - 45 sin a, y + 45 cos a)."""
    x, y, angle = block.astype(np.float64).T
    return np.stack([x - T_CENTRE_OFFSET * np.sin(angle), y + T_CENTRE_OFFSET * np.cos(angle)], axis=1)


def download(raw_dir: Path) -> Path:
    """The cached zip in `raw_dir`, downloaded from the original URL the first time."""
    path = Path(raw_dir) / "pusht.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(path)
    return path
