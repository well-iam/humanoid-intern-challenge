"""Recording setups: what changes with the room, the camera mount and the object.

A take is recorded in one setup. CARPET is the real takes' setup and the default everywhere; WOOD is the
morning test take of 16 Oct, kept so its numbers can be reproduced. Both use the same printed marker.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Setup:
    # The table square in the marker's own frame as printed (origin at its top-left corner, x toward its right
    # edge, y toward its bottom edge, cm), corner by corner in view order: top-left, top-right, bottom-right,
    # bottom-left. The view keeps the camera's orientation: the hand comes in from the bottom.
    square: tuple[tuple[float, float], ...]
    camera_height: float  # cm, lens above the table
    object_size: tuple[float, float, float]  # cm: width, length (along table y at angle 0), height
    object_pixels: Callable[[np.ndarray], np.ndarray]  # RGB image (..., 3) uint8 -> bool: the object's top


def _white(rgb: np.ndarray) -> np.ndarray:
    """The white box's top face: every channel bright, and grey rather than coloured. Its side faces (min channel
    about 100-135), the carpet's pale flecks (up to about 155) and skin (saturated) are not."""
    rgb = rgb.astype(np.int16)
    return (rgb.min(axis=-1) > 160) & (np.ptp(rgb, axis=-1) < 40)


def _dark_not_orange(rgb: np.ndarray) -> np.ndarray:
    """The dark wallet on light wood: every channel dark, and red at most a little above blue (the wood and the
    wallet's shadow on it are orange)."""
    rgb = rgb.astype(np.int16)
    return (rgb.max(axis=-1) < 100) & (rgb[..., 0] - rgb[..., 2] < 25)


# 16 Oct afternoon, the real takes: a white box 17.8 x 5.8 x 3 cm (ruler; rigid) on a dark, textured carpet, the
# camera 71 cm up. The square starts 5 cm beyond the marker's top edge and is shifted 14 cm toward its left
# edge (4 cm more than WOOD's, toward the hand side, to fit the view with a margin all round).
CARPET = Setup(
    square=((20.0, -55.0), (20.0, -5.0), (-30.0, -5.0), (-30.0, -55.0)),
    camera_height=71.0,
    object_size=(5.8, 17.8, 3.0),
    object_pixels=_white,
)

# 16 Oct morning test take: a dark wallet on light wood, the camera 77 cm up. The wallet is not rigid and its
# sides bulge, so the camera sees about 13 x 11 cm (ruler: 11.5 x 10 x 2).
WOOD = Setup(
    square=((24.0, -55.0), (24.0, -5.0), (-26.0, -5.0), (-26.0, -55.0)),
    camera_height=77.0,
    object_size=(11.0, 13.0, 2.0),
    object_pixels=_dark_not_orange,
)
