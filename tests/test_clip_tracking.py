"""Seam B for clip tracking: a take (phone video) -> tracked clips -> episodes.

Slow: runs on the 3-clip test take in $PWM_DATA/clips/test-take (else ./data/clips/test-take). That take is a
setup check only, never project data. Expected values come from the take by eye, from a ruler, or from ticket
decisions (16 Oct), never from the tracker itself.
"""

import dataclasses

import numpy as np
import pytest

from pwm import stages
from pwm.datasets import tracking
from pwm.episodes import save_episodes

pytestmark = pytest.mark.slow

TEST_TAKE = "test-take/VID_20261016_095359.mp4"


@pytest.fixture(scope="module")
def tracked():
    video = stages.data_root() / "clips" / TEST_TAKE
    if not video.exists():
        pytest.skip(f"needs the test take at {video}")
    return tracking.track_take(video, models=stages.data_root() / "models")


@pytest.fixture(scope="module")
def episodes(tracked):
    return tracking.to_episodes(tracked)


def test_the_a4_sheet_measures_its_printed_size_so_the_pixel_to_cm_error_is_known(tracked):
    # The sheet under the marker is A4, 29.7 x 21.0 cm. It lies flat on the table, so its straightened size
    # measures the whole conversion: marker, homography and camera.
    for clip in tracked:
        np.testing.assert_allclose(clip.sheet, [29.7, 21.0], atol=0.5, err_msg=clip.name)


def test_the_wallets_seen_footprint_is_the_same_in_every_clip_and_about_13_by_11_cm(tracked):
    # The ruler says 11.5 x 10 x 2 cm, but the wallet is not rigid and its sides bulge, so the camera sees about
    # 13 x 11 cm (decided 16 Oct). Sizes are (width, length): across and along its long side.
    sizes = np.array([clip.object_size for clip in tracked])
    assert np.ptp(sizes, axis=0).max() < 0.5, sizes
    np.testing.assert_allclose(sizes, [[11.0, 13.0]] * len(tracked), atol=0.5)


def test_the_palm_is_tracked_from_the_hands_entry_to_its_exit_without_a_long_gap(tracked):
    # By eye the hand comes into the square 2.4-3 s after each clap and pushes for about 2.5 s.
    for clip in tracked:
        assert 2.2 <= clip.entered <= 3.2, clip.name
        assert clip.left - clip.entered > 2.0, clip.name
        assert clip.palm_gaps == [], clip.name  # gaps longer than 0.25 s


def test_no_gaps_remain_in_the_palm_and_object_tracks_after_filling(episodes):
    assert len(episodes) == 3
    for ep in episodes:
        assert np.isfinite(ep.pusher_pos).all() and np.isfinite(ep.object_pos).all(), ep.name
        assert np.isfinite(ep.object_angle).all() and np.isfinite(ep.hand_direction).all(), ep.name


def test_frames_before_the_hand_comes_in_are_kept_with_the_palm_held_and_flagged_as_not_seen(episodes):
    for ep in episodes:
        assert ep.timestamps[0] == 0  # the clap
        entered = int(np.argmax(ep.pusher_seen))
        assert 2.2 <= ep.timestamps[entered] <= 3.3, ep.name
        assert not ep.pusher_seen[:entered].any()
        held = ep.pusher_pos[:entered]
        assert (held == held[0]).all(), ep.name
        # Held at the first tracked point, a video frame or two before this frame: the hand moves little.
        assert np.linalg.norm(held[0] - ep.pusher_pos[entered]) < 2.0, ep.name


def test_each_episode_ends_when_the_hand_leaves_the_view(episodes):
    # By eye the hand is gone 6.0 (at the 6 s cap), 5.2 and 5.7 s after the clap. MediaPipe loses a hand that
    # is half out of view and blurred, a little earlier.
    for ep, gone in zip(episodes, [6.0, 5.2, 5.7]):
        assert gone - 0.6 <= ep.timestamps[-1] <= gone + 0.1, ep.name
        assert ep.pusher_seen[-1], ep.name


def test_clips_where_the_wallet_moves_more_than_1_cm_are_pushes_the_others_near_misses(episodes):
    # By eye: clips 1 and 2 push the wallet across the table, in clip 3 the hand passes beside it.
    assert [ep.kind for ep in episodes] == ["push", "push", "near-miss"]


def test_the_wallets_angle_is_0_when_its_long_side_lies_along_table_y_and_follows_its_turns(episodes):
    def off_upright(angle):  # degrees from the nearest upright pose (a rectangle repeats every half turn)
        return abs((np.degrees(angle) + 90) % 180 - 90)

    # By eye the wallet ends clip 1 and starts clip 2 upright: long side along table y (up the view).
    assert off_upright(episodes[0].object_angle[-1]) < 10
    assert off_upright(episodes[1].object_angle[0]) < 10
    # In clip 1 it starts turned about a quarter of a right angle and the push straightens it.
    assert 15 < off_upright(episodes[0].object_angle[0]) < 40


def test_the_hand_direction_points_up_the_view_as_the_hand_pushes_from_the_bottom(episodes):
    # By eye the hand comes in from the bottom of the view, fingers first, and pushes up (toward -y).
    for ep in episodes:
        assert (ep.hand_direction[ep.pusher_seen, 1] < -0.7).all(), ep.name


def test_the_test_take_runs_through_the_toy_encoder_to_a_static_baseline_report(episodes, tmp_path):
    # The probe picks its ridge strength on every 5th training episode, so it needs at least 5; the test take
    # has 3 clips. Copies stand in for a full take: clips 1 and 3 train (4 copies each), clip 2 is held out.
    stand_in = [
        dataclasses.replace(ep, name=f"{ep.name}-{c}", split="held-out" if i == 1 else "train")
        for i, ep in enumerate(episodes)
        for c in range(1 if i == 1 else 4)
    ]
    save_episodes(stand_in, tmp_path / "clips-test-take" / "episodes")
    stages.embed("clips-test-take", "toy", root=tmp_path)
    report = stages.evaluate("clips-test-take", "toy", root=tmp_path)

    assert report.unit == "cm" and report.step == 0.25 and report.move_threshold == 1.0
    assert report.n_windows > 0 and report.n_moving[-1] > 0  # the held-out clip is a push
    static = report.errors["static baseline"]
    # Still windows flatter the static baseline: it looks better overall than on moving windows.
    assert all(moving > overall for moving, overall in zip(static["moving"], static["all"]))
    assert (tmp_path / "clips-test-take" / "toy" / "report.md").exists()
