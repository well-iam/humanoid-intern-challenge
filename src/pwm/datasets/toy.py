"""Toy adapter: a small synthetic pushing world, so the whole pipeline can be tested on a CPU.

A round pusher moves in a straight line past or into a square object, then stops. The
object moves only while the pusher touches it. Frames are 64 x 64 pixels and one toy unit
is one pixel. The object is drawn in the red channel and the pusher in the blue channel,
with exact area coverage at the edges.
"""

import numpy as np

from pwm.episodes import Episode

IMAGE_SIZE = 64
WORKSPACE = 64.0
UNIT = "toy units"
OBJECT_SIDE = 16.0
PUSHER_RADIUS = 5.0
FPS = 10
N_FRAMES = 40

STEP = 0.2  # seconds between frames the model sees
MOVE_THRESHOLD = 1.0  # toy units


def make_episodes(n_episodes: int = 60, n_held_out: int | None = None, seed: int = 0) -> list[Episode]:
    """Generate toy episodes. Every 4th one is a near-miss; the held-out ones are picked by the seed."""
    rng = np.random.default_rng(seed)
    if n_held_out is None:
        n_held_out = n_episodes // 3
    held_out = set(rng.permutation(n_episodes)[:n_held_out].tolist())
    episodes = []
    for i in range(n_episodes):
        kind = "near-miss" if i % 4 == 3 else "push"
        pusher_pos, object_pos = _simulate(rng, kind)
        episodes.append(
            Episode(
                name=f"toy-{i:03d}",
                frames=np.stack([_render(p, o) for p, o in zip(pusher_pos, object_pos)]),
                timestamps=np.arange(N_FRAMES) / FPS,
                pusher_pos=pusher_pos,
                object_pos=object_pos,
                workspace=WORKSPACE,
                unit=UNIT,
                split="held-out" if i in held_out else "train",
                kind=kind,
            )
        )
    return episodes


def build(raw_dir) -> list[Episode]:
    """Dataset-adapter entry point. The toy world has no raw source, so `raw_dir` is unused."""
    return make_episodes()


def _simulate(rng, kind: str) -> tuple[np.ndarray, np.ndarray]:
    obj = rng.uniform(28, 36, size=2)
    angle = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    if kind == "push":
        offset = rng.uniform(-5, 5)
    else:  # passes beside the object, never within touching distance
        offset = rng.choice([-1, 1]) * (PUSHER_RADIUS + OBJECT_SIDE / np.sqrt(2) + 2)
    # Pushes stay short enough (under 12 units) that the object never leaves the inner
    # 6 x 6 patches of the toy encoder, so training episodes cover every place it can be.
    pusher = obj - 21 * direction + offset * side
    speed = rng.uniform(0.55, 0.75)  # toy units per frame
    start = rng.integers(0, 6)
    n_moving = 26

    pusher_pos, object_pos = [], []
    for t in range(N_FRAMES):
        if start <= t < start + n_moving:
            for _ in range(10):  # small substeps keep the pusher from sinking into the object
                pusher = pusher + direction * speed / 10
                obj = _resolve_contact(pusher, obj)
        pusher_pos.append(pusher)
        object_pos.append(obj)
    return np.array(pusher_pos), np.array(object_pos)


def _resolve_contact(pusher: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """If the pusher overlaps the square, slide the square out along the contact normal."""
    half = OBJECT_SIDE / 2
    closest = np.clip(pusher, obj - half, obj + half)
    gap = closest - pusher
    dist = np.linalg.norm(gap)
    if dist >= PUSHER_RADIUS:
        return obj
    return obj + gap / dist * (PUSHER_RADIUS - dist)


_SUB = 4  # supersampling for the round pusher
_FINE = (np.arange(IMAGE_SIZE * _SUB) + 0.5) / _SUB


def _render(pusher: np.ndarray, obj: np.ndarray) -> np.ndarray:
    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    frame[..., 0] = np.round(255 * _square_coverage(obj))
    frame[..., 2] = np.round(255 * _disc_coverage(pusher))
    return frame


def _square_coverage(centre: np.ndarray) -> np.ndarray:
    """Exact fraction of each pixel covered by the axis-aligned square. Rows are y, columns x."""
    pixels = np.arange(IMAGE_SIZE)
    lo, hi = centre - OBJECT_SIDE / 2, centre + OBJECT_SIDE / 2
    cover_x = np.clip(np.minimum(hi[0], pixels + 1) - np.maximum(lo[0], pixels), 0, 1)
    cover_y = np.clip(np.minimum(hi[1], pixels + 1) - np.maximum(lo[1], pixels), 0, 1)
    return np.outer(cover_y, cover_x)


def _disc_coverage(centre: np.ndarray) -> np.ndarray:
    inside = (_FINE[None, :] - centre[0]) ** 2 + (_FINE[:, None] - centre[1]) ** 2 <= PUSHER_RADIUS**2
    return inside.reshape(IMAGE_SIZE, _SUB, IMAGE_SIZE, _SUB).mean(axis=(1, 3))
