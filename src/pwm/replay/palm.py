"""My palm paths -> pushes for the Panda replay (ticket 15, option A of 15 Oct).

The palm position is the centre of the palm, but the hand touches the object with its fingertips or its
edge, about 12 cm further on. The replay therefore moves each clip's whole pusher path by one vector, the
contact shift, before and after contact alike, so the gripper's path stays smooth. The world model keeps
the unshifted palm position; the shift is for the replay only.

- Contact frame: one frame (0.125 s) before the object is first more than STARTS_MOVING from where it started.
- Own shift: from the palm to the nearest point of the object's edge at the contact frame.
- Which shift: a push gets its own shift only if the hand was found at least LATE_HAND before the object
  starts moving (else the tracking report flags "hand found late", and the palm at contact is unknown or only
  just found). Every other clip (those pushes, and the near-misses, which have no contact) gets the median of
  the own shifts. The hand comes in from every side, so the median is taken in the hand's frame (ahead along
  the hand's direction at contact, and to its side) and turned to the clip's own hand direction (its median
  over the frames where the hand is found).
- Gripper: the shift brings the palm to the object's edge, where the hand touched it. The Panda's closed fingers
  reach GRIPPER_HALF beyond their centre, so the replayed path stops that much short (measured along the shift),
  so that the fingers' face, not their centre, is at the edge at contact. Still one vector for the whole path.
- Start: before the hand is found, the palm is only held at its first tracked point, so a replay starts at the
  first frame where the hand is found, with the object at its tracked pose in that frame.
"""

from dataclasses import dataclass

import numpy as np

from pwm.datasets import setups, tracking
from pwm.episodes import Episode
from pwm.replay import Push

STARTS_MOVING = tracking.STARTS_MOVING  # cm
BACK = 1  # frames from the object's first move back to the contact frame
LATE_HAND = tracking.LATE_HAND  # seconds
GRIPPER_HALF = (1.0, 1.9)  # cm: the Panda's closed fingers from above, half their size along table x and y


@dataclass(frozen=True)
class Shift:
    vector: np.ndarray  # (2,) table cm, added to every palm position of the clip
    own: bool  # measured on this clip; else the median of the pushes' own shifts


def starts_moving(ep: Episode) -> int | None:
    """The first frame where the object is more than STARTS_MOVING from where it started, or None."""
    away = np.linalg.norm(ep.object_pos - ep.object_pos[0], axis=1) > STARTS_MOVING
    return int(np.argmax(away)) if away.any() else None


def contact_frame(ep: Episode) -> int | None:
    """The frame just before the object starts moving, or None if it never does."""
    moving = starts_moving(ep)
    return None if moving is None else max(moving - BACK, 0)


def own_shift(ep: Episode, size=setups.CARPET.object_size) -> np.ndarray | None:
    """The clip's own contact shift (2,) in table cm: palm -> nearest point of the object's edge at the contact
    frame. None if the object never moves."""
    c = contact_frame(ep)
    if c is None:
        return None
    corners = tracking.object_corners(ep.object_pos[c], ep.object_angle[c], size)
    return _nearest_on_outline(ep.pusher_pos[c], corners) - ep.pusher_pos[c]


def in_time(ep: Episode) -> bool:
    """Whether the hand was found at least LATE_HAND before the object starts moving."""
    moving = starts_moving(ep)
    if moving is None or not ep.pusher_seen.any():
        return False
    return ep.timestamps[np.argmax(ep.pusher_seen)] <= ep.timestamps[moving] - LATE_HAND + 1e-9


def shifts(episodes: list[Episode], size=setups.CARPET.object_size) -> dict[str, Shift]:
    """Each clip's contact shift, by name."""
    own = {ep.name: own_shift(ep, size) for ep in episodes if ep.kind == "push" and in_time(ep)}
    assert own, "no push with the hand found in time: no shift to take the median of"
    by_name = {ep.name: ep for ep in episodes}
    in_hand_frame = [_to_hand_frame(v, by_name[name].hand_direction[contact_frame(by_name[name])])
                     for name, v in own.items()]
    median = np.median(in_hand_frame, axis=0)
    result = {}
    for ep in episodes:
        if ep.name in own:
            result[ep.name] = Shift(own[ep.name], True)
        else:
            direction = np.median(ep.hand_direction[ep.pusher_seen], axis=0)
            result[ep.name] = Shift(_from_hand_frame(median, direction / np.linalg.norm(direction)), False)
    return result


def _to_hand_frame(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """(ahead, side) of a table vector, for a unit hand direction; side is the direction turned a quarter."""
    side = np.array([-direction[1], direction[0]])
    return np.array([vector @ direction, vector @ side])


def _from_hand_frame(ahead_side: np.ndarray, direction: np.ndarray) -> np.ndarray:
    side = np.array([-direction[1], direction[0]])
    return ahead_side[0] * direction + ahead_side[1] * side


def clip_push(ep: Episode, shift: np.ndarray, size=setups.CARPET.object_size, gripper=GRIPPER_HALF) -> Push:
    """The push the Panda replays: the palm path moved by `shift`, less the gripper's half-size along it, from the
    first frame where the hand is found. gripper=(0, 0) moves the gripper's centre exactly by `shift`."""
    first = int(np.argmax(ep.pusher_seen))
    along = shift / np.linalg.norm(shift)
    standoff = gripper[0] * abs(along[0]) + gripper[1] * abs(along[1])  # the fingers' reach toward the edge
    return Push(path=ep.pusher_pos[first:] + shift - standoff * along, timestamps=ep.timestamps[first:],
                start=np.array([*ep.object_pos[first], ep.object_angle[first]]), shape="box", unit=ep.unit,
                workspace=ep.workspace, size=tuple(size))


def _nearest_on_outline(point: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The point of a polygon's outline (corners (N, 2) in order) nearest to `point`."""
    a, b = corners, np.roll(corners, -1, axis=0)
    t = np.clip(np.einsum("ij,ij->i", point - a, b - a) / np.einsum("ij,ij->i", b - a, b - a), 0, 1)
    candidates = a + t[:, None] * (b - a)
    return candidates[np.argmin(np.linalg.norm(candidates - point, axis=1))]
