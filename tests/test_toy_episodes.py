"""Seam B for the toy adapter: generated world -> episodes in the shared format."""

import numpy as np

from pwm.datasets import toy
from pwm.episodes import load_episodes, save_episodes


def test_same_seed_gives_the_same_episodes_and_split():
    a, b = toy.make_episodes(n_episodes=8, seed=3), toy.make_episodes(n_episodes=8, seed=3)
    assert [ep.split for ep in a] == [ep.split for ep in b]
    assert all(np.array_equal(x.frames, y.frames) and np.array_equal(x.object_pos, y.object_pos) for x, y in zip(a, b))


def test_object_moves_only_while_the_pusher_touches_it():
    episodes = toy.make_episodes(n_episodes=12, seed=0)
    # The pusher touches the square object only if its centre is within
    # pusher radius + half the square's diagonal of the object centre.
    touch_reach = toy.PUSHER_RADIUS + toy.OBJECT_SIDE / np.sqrt(2) + 0.5

    moved_any = False
    for ep in episodes:
        moved = np.linalg.norm(np.diff(ep.object_pos, axis=0), axis=1) > 1e-9
        gap = np.linalg.norm(ep.pusher_pos[1:] - ep.object_pos[1:], axis=1)
        assert np.all(gap[moved] <= touch_reach), ep.name
        if ep.kind == "near-miss":
            assert not moved.any(), ep.name
        moved_any |= moved.any()
    assert moved_any


def test_episodes_can_be_loaded_without_their_frames(tmp_path):
    # Most stages need only the tracks: a run folder of real clips holds gigabytes of frames.
    episodes = toy.make_episodes(n_episodes=4, seed=0)
    save_episodes(episodes, tmp_path)

    tracks = load_episodes(tmp_path, frames=False)

    for ep, loaded in zip(episodes, tracks):
        assert loaded.name == ep.name and loaded.split == ep.split and loaded.kind == ep.kind
        np.testing.assert_array_equal(loaded.object_pos, ep.object_pos)
        np.testing.assert_array_equal(loaded.pusher_pos, ep.pusher_pos)
        assert len(loaded.frames) == len(ep.frames) and loaded.frames.size == 0
