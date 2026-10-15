"""Seam B for the Push-T adapter: the downloaded Diffusion Policy file -> episodes.

Slow: needs the 31 MB download, cached in $PWM_DATA/pusht/raw (else ./data/pusht/raw).
"""

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from pwm import stages
from pwm.datasets import DATASETS, pusht
from pwm.windows import make_windows

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def raw_dir():
    return stages.data_root() / "pusht" / "raw"


@pytest.fixture(scope="module")
def episodes(raw_dir):
    return pusht.build(raw_dir)


@pytest.fixture(scope="module")
def raw(raw_dir, episodes):
    """The keypoints and agent states straight from the file, read independently of the adapter."""
    group = zarr.open_group(ZipStore(raw_dir / "pusht.zip", mode="r"), mode="r", path=pusht.ZARR_PATH)
    return group["data/keypoint"][:], group["data/state"][:], group["meta/episode_ends"][:]


def _rows(ep, episode_ends):
    index = int(ep.name.removeprefix("pusht-"))
    start = 0 if index == 0 else episode_ends[index - 1]
    return slice(start, episode_ends[index])


def test_split_is_100_training_and_50_held_out_episodes(episodes):
    splits = [ep.split for ep in episodes]
    assert splits.count("train") == 100 and splits.count("held-out") == 50
    assert len({ep.name for ep in episodes}) == 150


def test_object_position_is_the_t_centre(episodes, raw):
    # The keypoint mean is a fixed point of the T, 4.7 units from its centre. In the first
    # few frames after a reset the stored keypoints and pose disagree slightly (up to 5.6
    # units in 16 of the 206 episodes), so those frames are left out.
    settled = 4
    keypoints, _, ends = raw
    for ep in episodes:
        keypoint_mean = keypoints[_rows(ep, ends)].mean(axis=1)
        gap = np.linalg.norm(ep.object_pos - keypoint_mean, axis=1)
        assert gap[settled:].max() < 5, ep.name


def test_pusher_actions_are_the_observed_agent_displacements(episodes, raw):
    _, state, ends = raw
    windows = make_windows(episodes, DATASETS["pusht"].step, future=2)
    for i, ep in enumerate(episodes):
        agent = state[_rows(ep, ends), :2]
        rows = windows.episode == i
        frames = windows.frames[rows]
        np.testing.assert_allclose(windows.actions[rows], agent[frames[:, 1:]] - agent[frames[:, :-1]], atol=1e-4)
        # Step 0.5 s at 10 Hz is stride 5, and every starting offset is used.
        assert np.all(np.diff(frames, axis=1) == 5)
        assert set(frames[:, 0] % 5) == set(range(5))
