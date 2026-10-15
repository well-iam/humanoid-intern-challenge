"""Seam C for the Push-T judge: pusher path + object start pose -> final object pose.

Slow: runs Push-T's own simulator, and the replay check needs the Push-T download.
"""

import numpy as np
import pytest

from pwm import stages
from pwm.datasets import pusht
from pwm.episodes import load_episodes
from pwm.replay import Push, get_replay

pytestmark = pytest.mark.slow


def test_a_pusher_that_never_touches_the_t_leaves_it_where_it_started():
    start = np.array([300.0, 250.0, 0.7])  # T centre x, y and angle
    path = np.stack([np.linspace(40, 470, 21), np.full(21, 40.0)], axis=1)  # along the top wall
    push = Push(path=path, timestamps=np.arange(21) * 0.1, start=start, shape="T", unit="Push-T units")

    result = get_replay("pusht").replay(push, frame_size=96)

    np.testing.assert_allclose(result.pusher, path, atol=0.5)  # the pusher is on the path, not lagging
    np.testing.assert_allclose(result.final, start, atol=1e-6)
    np.testing.assert_allclose(result.poses, np.tile(start, (21, 1)), atol=1e-6)
    assert result.frames.shape == (21, 96, 96, 3) and result.frames.dtype == np.uint8


def test_replaying_recorded_episodes_reproduces_their_final_t_position():
    # The judge's noise floor: replay each held-out episode's pusher path from its recorded
    # start and compare the final T centre with the recorded one. It must be under 4% of
    # the workspace (Gate 1's bar for the probe), or the judge cannot rank what the probe sees.
    root = stages.data_root() / "pusht"
    episodes = [ep for ep in load_episodes(root / "episodes") if ep.split == "held-out"]
    poses = pusht.block_poses(root / "raw")
    judge = get_replay("pusht")

    errors, moved = [], []
    for ep in episodes:
        np.testing.assert_allclose(poses[ep.name][:, :2], ep.object_pos, atol=1e-3)
        push = Push(path=ep.pusher_pos, timestamps=ep.timestamps, start=poses[ep.name][0],
                    shape="T", unit=ep.unit, workspace=ep.workspace)
        final = judge.replay(push).final
        errors.append(np.linalg.norm(final[:2] - ep.object_pos[-1]))
        moved.append(np.linalg.norm(ep.object_pos[-1] - ep.object_pos[0]))

    # The replayed frames look like the recorded ones, from the very first frame.
    frames = judge.replay(push, frame_size=96).frames
    pixel_gap = np.abs(frames.astype(int) - ep.frames.astype(int)).mean(axis=(1, 2, 3))
    assert pixel_gap.max() < 1, pixel_gap.argmax()

    assert len(errors) == 50
    assert np.median(errors) < 0.04 * pusht.WORKSPACE
    assert np.median(errors) < 0.1 * np.median(moved)
