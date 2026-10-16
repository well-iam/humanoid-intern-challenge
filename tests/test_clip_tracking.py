"""Seam B for clip tracking: a take (phone video) -> tracked clips -> episodes.

Slow: runs on the 3-clip test take of the real takes' setup (white box on the carpet, 16 Oct afternoon) in
$PWM_DATA/clips/test-take (else ./data/clips/test-take). That take is a setup check only, never project data.
Expected values come from the take by eye, from a ruler, or from ticket decisions, never from the tracker itself.
"""

import dataclasses

import numpy as np
import pytest

from pwm import stages
from pwm.datasets import setups, tracking
from pwm.episodes import save_episodes

pytestmark = pytest.mark.slow

TEST_TAKE = "test-take/VID_20261016_131201.mp4"
FINGERTIPS = [2.75, 2.25, 2.125]  # s after the clap: by eye, the fingertips first show at the frame's bottom edge
GONE = [5.625, 4.5, 4.75]  # s after the clap: by eye, the first frame with no part of the hand in view


@pytest.fixture(scope="module")
def tracked():
    video = stages.data_root() / "clips" / TEST_TAKE
    if not video.exists():
        pytest.skip(f"needs the test take at {video}")
    return tracking.track_take(video, models=stages.data_root() / "models", setup=setups.CARPET)


@pytest.fixture(scope="module")
def episodes(tracked):
    return tracking.to_episodes(tracked)


def test_the_a4_sheets_short_side_measures_its_printed_size_and_its_cut_long_side_is_left_out(tracked):
    # The sheet under the marker is A4, 29.7 x 21.0 cm, flat on the floor, so its straightened size measures the
    # whole conversion. On this layout its far end runs off the frame's right edge (check/<take>/layout.png),
    # so only its short side can be measured.
    for clip in tracked:
        assert np.isnan(clip.sheet[0]), clip.name
        assert abs(clip.sheet[1] - 21.0) < 0.5, clip.name


def test_the_boxs_top_face_measures_its_ruler_size_in_every_clip(tracked):
    # The box is rigid: 17.8 x 5.8 x 3 cm by ruler. After the height fix its top face is that size wherever it
    # lies in the view. Sizes are (width, length): across and along its long side.
    for clip in tracked:
        np.testing.assert_allclose(clip.object_size, [5.8, 17.8], atol=0.5, err_msg=clip.name)


def test_the_palm_is_tracked_from_the_hands_entry_to_its_exit_without_a_long_gap(tracked):
    # MediaPipe needs the palm in view, so it finds the hand a little after the fingertips show.
    for clip, fingertips, gone in zip(tracked, FINGERTIPS, GONE):
        assert fingertips <= clip.entered <= fingertips + 0.75, clip.name
        assert gone - 0.6 <= clip.left < gone, clip.name
        assert clip.palm_gaps == [], clip.name  # gaps longer than 0.25 s


def test_no_gaps_remain_in_the_palm_and_object_tracks_after_filling(episodes):
    assert len(episodes) == 3
    for ep in episodes:
        assert np.isfinite(ep.pusher_pos).all() and np.isfinite(ep.object_pos).all(), ep.name
        assert np.isfinite(ep.object_angle).all() and np.isfinite(ep.hand_direction).all(), ep.name


def test_frames_before_the_hand_comes_in_are_kept_with_the_palm_held_and_flagged_as_not_seen(episodes):
    for ep, fingertips in zip(episodes, FINGERTIPS):
        assert ep.timestamps[0] == 0  # the clap
        entered = int(np.argmax(ep.pusher_seen))
        assert fingertips <= ep.timestamps[entered] <= fingertips + 0.75, ep.name
        assert not ep.pusher_seen[:entered].any()
        held = ep.pusher_pos[:entered]
        assert (held == held[0]).all(), ep.name
        # Held at the first tracked point, a video frame or two before this frame: the hand moves little.
        assert np.linalg.norm(held[0] - ep.pusher_pos[entered]) < 2.0, ep.name


def test_each_episode_ends_when_the_hand_leaves_the_view(episodes):
    # MediaPipe loses a hand that is half out of view a little before the last finger leaves.
    for ep, gone in zip(episodes, GONE):
        assert gone - 0.6 <= ep.timestamps[-1] < gone, ep.name
        assert ep.pusher_seen[-1], ep.name


def test_clips_where_the_box_moves_more_than_1_cm_are_pushes_the_others_near_misses(episodes):
    # By eye: clips 1 and 2 push the box up the view, in clip 3 the hand passes left of it.
    assert [ep.kind for ep in episodes] == ["push", "push", "near-miss"]


def test_the_boxs_angle_is_0_when_its_long_side_lies_along_table_y_and_follows_its_turns(episodes):
    def angle(a):  # degrees in (-90, 90] (a rectangle repeats every half turn)
        return (np.degrees(a) + 90) % 180 - 90

    # By eye the box starts with its long side from upper left to lower right (negative angles): about 35
    # degrees below the horizontal in clip 1, steeper (about 50 degrees below) in clip 3.
    assert -65 < angle(episodes[0].object_angle[0]) < -45
    assert -50 < angle(episodes[2].object_angle[0]) < -30
    # The push in clip 1 turns it about 10 degrees toward upright.
    turned = abs(angle(episodes[0].object_angle[0])) - abs(angle(episodes[0].object_angle[-1]))
    assert 5 < turned < 20


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
    # Still windows flatter the static baseline: it looks better overall than on moving windows (at the horizons
    # that have any; the held-out push lasts under a second, so the shortest horizon may have none).
    pairs = zip(report.n_moving, static["moving"], static["all"])
    assert all(moving > overall for n, moving, overall in pairs if n)
    assert (tmp_path / "clips-test-take" / "toy" / "report.md").exists()


def test_the_morning_test_take_still_gives_its_numbers_in_its_own_setup():
    # The wallet on wood (16 Oct morning, setup WOOD): the A4 sheet whole, the wallet seen as about 11 x 13 cm in
    # every clip (decided 16 Oct), and by eye clips 1 and 2 push it, clip 3 passes beside it.
    video = stages.data_root() / "clips" / "test-take/VID_20261016_095359.mp4"
    if not video.exists():
        pytest.skip(f"needs the morning test take at {video}")
    tracked = tracking.track_take(video, models=stages.data_root() / "models", setup=setups.WOOD)
    sizes = np.array([clip.object_size for clip in tracked])
    np.testing.assert_allclose([clip.sheet for clip in tracked], [[29.7, 21.0]] * 3, atol=0.5)
    np.testing.assert_allclose(sizes, [[11.0, 13.0]] * 3, atol=0.5)
    assert np.ptp(sizes, axis=0).max() < 0.5
    assert [clip.kind for clip in tracked] == ["push", "push", "near-miss"]
