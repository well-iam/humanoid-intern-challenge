"""Windowing: cuts episodes into windows of context frames followed by future steps.

A window is `context + future` frames, one step apart, starting at any frame of an
episode (every starting offset is used). The last context frame is the current frame.
"""

from dataclasses import dataclass

import numpy as np

from pwm.episodes import Episode

CONTEXT = 3


@dataclass
class Windows:
    context: int
    episode: np.ndarray  # (N,) index into the episode list
    frames: np.ndarray  # (N, L) frame indices within that episode, L = context + future
    pusher_pos: np.ndarray  # (N, L, 2)
    object_pos: np.ndarray  # (N, L, 2)

    def __len__(self) -> int:
        return len(self.episode)

    @property
    def actions(self) -> np.ndarray:
        """Pusher actions (N, L - 1, 2): how far the pusher moves over each step."""
        return np.diff(self.pusher_pos, axis=1)

    def gather(self, per_episode: list[np.ndarray], rows=slice(None), cols=slice(None)) -> np.ndarray:
        """Per-frame values (e.g. frame embeddings) for windows `rows` at window positions `cols`."""
        return _gather(per_episode, self.episode[rows], self.frames[rows][:, cols])


def make_windows(episodes: list[Episode], step: float, future: int, context: int = CONTEXT) -> Windows:
    """Windows of `context` frames plus `future` steps from each episode; `step` is in seconds."""
    length = context + future
    episode_ids, frames = [], []
    for i, ep in enumerate(episodes):
        stride = _stride(ep, step)
        span = (length - 1) * stride
        for start in range(len(ep.frames) - span):
            episode_ids.append(i)
            frames.append(start + stride * np.arange(length))
    episode_ids = np.array(episode_ids, dtype=int)
    frames = np.array(frames, dtype=int).reshape(-1, length)
    return Windows(
        context=context,
        episode=episode_ids,
        frames=frames,
        pusher_pos=_gather([ep.pusher_pos for ep in episodes], episode_ids, frames),
        object_pos=_gather([ep.object_pos for ep in episodes], episode_ids, frames),
    )


def _stride(ep: Episode, step: float) -> int:
    """Number of frames per step. Frames must be evenly spaced and the step a multiple of their spacing."""
    spacing = np.median(np.diff(ep.timestamps))
    stride = int(round(step / spacing))
    assert stride >= 1 and np.isclose(stride * spacing, step, rtol=1e-3), (ep.name, step, spacing)
    return stride


def _gather(per_episode: list[np.ndarray], episode_ids: np.ndarray, frames: np.ndarray) -> np.ndarray:
    if len(episode_ids) == 0:
        return np.zeros((0, *frames.shape[1:], *per_episode[0].shape[1:]), per_episode[0].dtype)
    return np.stack([per_episode[e][f] for e, f in zip(episode_ids, frames)])
