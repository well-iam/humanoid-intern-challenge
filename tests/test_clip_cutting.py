"""Seam B for clip cutting: a take (phone video) -> clips cut at the claps, straightened to the table square.

Slow: runs on the 3-clip test take in $PWM_DATA/clips/test-take (else ./data/clips/test-take). That take is a
setup check only, never project data. Expected values below were read off the take by eye and ear (16 Oct).
"""

import av
import numpy as np
import pytest

from pwm import stages
from pwm.datasets import clips

pytestmark = pytest.mark.slow

TEST_TAKE = "test-take/VID_20261016_095359.mp4"


@pytest.fixture(scope="module")
def take():
    video = stages.data_root() / "clips" / TEST_TAKE
    if not video.exists():
        pytest.skip(f"needs the test take at {video}")
    return clips.cut_take(video)


def test_the_three_claps_are_found(take):
    # Heard at 1.57, 12.47 and 22.99 s. A click when the phone was stopped (29.7 s) is not a clap.
    np.testing.assert_allclose(take.claps, [1.57, 12.47, 22.99], atol=0.05)


def test_each_clip_is_6_s_of_frames_picked_by_timestamp_at_8_fps(take):
    assert [len(clip.frames) for clip in take.clips] == [48, 48, 48]
    for clip in take.clips:
        np.testing.assert_allclose(clip.timestamps, np.arange(48) / 8)
        # The phone records about 30 fps with uneven gaps (31-36 ms): each frame is the one nearest its time.
        assert np.abs(clip.source_times - (clip.clap + clip.timestamps)).max() < 0.018


def test_the_marker_is_found_in_every_frame_of_the_hand_free_second_after_each_clap(take):
    # The hand comes into view 1.2-1.3 s after each clap on the test take.
    for clip in take.clips:
        assert clip.hand_free_frames >= 28  # about 30 fps
        assert clip.hand_free_found == clip.hand_free_frames


def test_the_marker_corners_drift_less_than_1_px_within_each_clip(take):
    for clip in take.clips:
        assert np.nanmax(clip.drift_px) < 1, clip.name
        assert not clip.drifted


def test_a_camera_step_between_clips_is_flagged_not_failed(take):
    # The camera moved once, between clips 1 and 2, by about 2 px: the wood grain away from the marker
    # shifted with it. Each clip has its own homography, so the step is only reported.
    assert [clip.camera_moved for clip in take.clips] == [False, True, False]
    assert 1.5 < take.clips[1].step_px < 3


def test_the_straightened_view_has_the_sheet_on_its_right_as_the_camera_sees_it(take):
    # In the camera image the sheet lies right of the push area, level with its upper half; the hand comes
    # in from the bottom. The view keeps that orientation, and the marker stays out of it.
    for clip in take.clips:
        assert clip.frames.shape[1:] == (clips.SIZE, clips.SIZE, 3)
        marker = clip.to_cm(clip.corners)
        assert (marker[:, 0] > clips.WORKSPACE + 4).all()
        assert 0 < marker[:, 1].mean() < clips.WORKSPACE / 2


def test_a_point_in_the_straightened_frames_is_where_to_cm_puts_it(take):
    # The wallet's centre, found once in the raw video and mapped with to_cm, and once in the straightened
    # frame, where pixel (u, v) covers table cm [u, u + 1) * 50 / SIZE.
    clip = take.clips[0]
    with av.open(str(stages.data_root() / "clips" / TEST_TAKE)) as container:
        raw = next(f for f in container.decode(video=0) if f.time >= clip.source_times[0] - 1e-3)
        raw = raw.to_ndarray(format="rgb24")
    ys, xs = np.nonzero(raw.max(axis=-1) < 60)  # the wallet and the marker are the only near-black things
    cm = clip.to_cm(np.stack([xs, ys], axis=1).astype(float))
    in_view = ((cm >= 0) & (cm <= clips.WORKSPACE)).all(axis=1)
    raw_centre = cm[in_view].mean(axis=0)

    vs, us = np.nonzero(clip.frames[0].max(axis=-1) < 60)
    view_centre = (np.stack([us, vs], axis=1).mean(axis=0) + 0.5) * clips.WORKSPACE / clips.SIZE
    assert np.linalg.norm(view_centre - raw_centre) < 0.3
