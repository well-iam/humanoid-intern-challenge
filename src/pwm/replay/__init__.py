"""Replay simulators: a pusher path and the object's start pose -> where the object ends up.

Every simulator takes the same `Push` and returns the same `Replayed`, so any data source
(Push-T episodes in Push-T units, my clips in cm) can drive any simulator that supports
its object shape:

- "pusht": Push-T's own pymunk simulator (gym-pusht), the judge of the ranking test.
- "panda": a Panda in MuJoCo (ticket 06).

Poses are (x, y, angle): the object position (the centre of the object, as in the
episodes) in table units, and the angle in radians, counter-clockwise in the table frame
with y pointing down the image, as in Push-T. Angles are not wrapped, so they can be
interpolated.

Object shapes:
- "T": Push-T's T. In Push-T units, a 120 x 30 crossbar and a 30 x 90 stem. At angle 0 the
  crossbar is on top (small y) and the stem points to +y; the centre is the mean of the two
  bar centres, 45 units from the crossbar's outer edge.
- "box": a box of `size` = (width along x, length along y, height) at angle 0.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Push:
    path: np.ndarray  # (T, 2) pusher positions, table units
    timestamps: np.ndarray  # (T,) seconds, increasing
    start: np.ndarray  # (3,) object pose at timestamps[0]
    shape: str  # "T" or "box"
    unit: str  # table unit, e.g. "Push-T units" or "cm"
    workspace: float = 512.0  # width of the square workspace, table units
    size: tuple[float, ...] = ()  # box only: (width, length, height), table units


@dataclass
class Replayed:
    final: np.ndarray  # (3,) object pose at the last timestamp
    poses: np.ndarray  # (T, 3) object pose at each timestamp
    pusher: np.ndarray  # (T, 2) where the simulated pusher really was at each timestamp
    frames: np.ndarray | None  # (T, S, S, 3) uint8 top-down frames, if asked for


def get_replay(name: str):
    """A replay simulator: `.replay(push, frame_size=None) -> Replayed`. Frames are rendered only
    when `frame_size` (the side of the square frames in pixels) is given."""
    if name == "pusht":
        from pwm.replay.pusht import PushTJudge

        return PushTJudge()
    raise ValueError(f"unknown replay simulator: {name}")
