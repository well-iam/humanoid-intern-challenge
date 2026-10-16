"""Tracked clips -> episodes: the labels and the held-out split, on made-up tracks (fast).

The real-video side of the clip adapter is Seam B in test_clip_tracking.py (slow). The split cannot be checked
there: the test take has only 3 clips, and Friday's takes are recorded after this code.
"""

import numpy as np

from pwm.datasets import setups, tracking
from pwm.episodes import load_episodes, save_episodes


def made_up_clip(name: str, moved: float, n: int = 12, entered: float = 0.25) -> tracking.TrackedClip:
    """A tracked clip whose object slides `moved` cm along x from 0.5 to 1 s after the clap, the hand being found
    from `entered` s."""
    t = np.arange(n) / 8
    object_pos = np.stack([10 + moved * np.clip(t - 0.5, 0, 0.5) / 0.5, np.full(n, 20.0)], axis=1)
    return tracking.TrackedClip(
        name=name,
        frames=np.zeros((n, 8, 8, 3), np.uint8),
        timestamps=t,
        sheet=np.array(tracking.SHEET),
        drift_px=0.1,
        object_pos=object_pos,
        object_angle=np.zeros(n),
        object_seen=np.ones(n, bool),
        object_size=np.array(setups.CARPET.object_size[:2]),
        palm=object_pos + [0, 9],
        palm_seen=t >= entered,
        hand_direction=np.tile([0.0, -1.0], (n, 1)),
        entered=entered,
        left=t[-1],
        kept=n,
    )


def friday(n_push: int = 30, n_near_miss: int = 10, takes=("VID_0", "VID_1", "VID_2")) -> list[tracking.TrackedClip]:
    """Clips of a few takes, pushes and near-misses mixed, in no particular order."""
    kinds = ["push"] * n_push + ["near-miss"] * n_near_miss
    order = np.random.default_rng(7).permutation(len(kinds))
    per_take = -(-len(kinds) // len(takes))
    return [made_up_clip(f"{takes[k // per_take]}-{k % per_take + 1:02d}", 5.0 if kinds[i] == "push" else 0.3)
            for k, i in enumerate(order)]


# Ticket 13's held-out clips (its tracking report), from the first three takes (14:13-14:16).
FIRST_HELD_OUT = {
    "VID_20261016_141304-03", "VID_20261016_141304-05", "VID_20261016_141304-11", "VID_20261016_141455-02",
    "VID_20261016_141455-06", "VID_20261016_141455-09", "VID_20261016_141631-03", "VID_20261016_141631-08",
}


def first_takes() -> list[tracking.TrackedClip]:
    """The first three takes as ticket 13 cut them: 11 + 10 + 9 clips, 22 pushes and 8 near-misses."""
    near_misses = {"VID_20261016_141631-03", "VID_20261016_141631-08", "VID_20261016_141455-04",
                   "VID_20261016_141631-01", "VID_20261016_141631-02", "VID_20261016_141631-04",
                   "VID_20261016_141631-05", "VID_20261016_141631-06"}
    names = [f"VID_20261016_{take}-{k:02d}" for take, n in [("141304", 11), ("141455", 10), ("141631", 9)]
             for k in range(1, n + 1)]
    return [made_up_clip(name, 0.3 if name in near_misses else 5.0) for name in names]


def test_the_object_moving_more_than_1_cm_makes_a_push():
    episodes = tracking.to_episodes([made_up_clip("a", 1.5), made_up_clip("b", 0.6)])
    assert [ep.kind for ep in episodes] == ["push", "near-miss"]


def test_the_first_takes_keep_their_8_held_out_clips_and_the_new_takes_add_4_pushes_and_6_near_misses():
    new = friday(n_push=40, n_near_miss=30, takes=("VID_20261016_163613", "VID_20261016_165020"))
    clips = first_takes() + new
    first = tracking.to_episodes(clips)
    again = tracking.to_episodes(clips[::-1])  # the order the takes are read in does not matter

    held_out = {ep.name: ep.kind for ep in first if ep.split == "held-out"}
    from_first_takes = {name for name in held_out if not name.startswith(("VID_20261016_163613", "VID_20261016_165020"))}
    assert from_first_takes == FIRST_HELD_OUT
    added = [kind for name, kind in held_out.items() if name not in FIRST_HELD_OUT]
    assert added.count("push") == 4 and added.count("near-miss") == 6
    assert [(ep.name, ep.split) for ep in first] == [(ep.name, ep.split) for ep in again]


def test_an_excluded_clip_is_left_out_before_the_draw():
    clips = friday(n_push=20, n_near_miss=20)
    kept = tracking.to_episodes(clips, excluded={"VID_0-01": "the hand moves the box back"})
    assert "VID_0-01" not in [ep.name for ep in kept]
    assert len(kept) == len(clips) - 1
    assert sum(ep.split == "held-out" for ep in kept) == 10


def test_a_clip_ends_when_the_hand_leaves_so_a_return_to_reset_the_object_is_not_a_push():
    # 3 s at 8 fps. The object moves from 0.5 to 1 s (made_up_clip); the hand is seen at `seen` frames.
    def clip(name, seen):
        c = made_up_clip(name, 5.0, n=24, entered=0.0)
        c.palm_seen = np.zeros(24, bool)
        c.palm_seen[seen] = True
        return c

    # Near-miss, then the hand is gone for 1 s, then it comes back and moves the object (a reset).
    reset = clip("reset", np.r_[0:3, 11:24])
    # The hand is lost for half a second mid-push: the same visit.
    dropout = clip("dropout", np.r_[0:6, 10:24])
    episodes = {ep.name: ep for ep in tracking.to_episodes([reset, dropout])}

    assert episodes["reset"].kind == "near-miss"
    assert episodes["reset"].timestamps[-1] == 0.25  # the last frame of the first visit
    assert episodes["dropout"].kind == "push" and len(episodes["dropout"].timestamps) == 24


def test_a_small_set_holds_out_at_most_half_of_each_kind():
    # The 3-clip test take: 2 pushes and a near-miss.
    episodes = tracking.to_episodes(friday(n_push=2, n_near_miss=1))
    held_out = [ep.kind for ep in episodes if ep.split == "held-out"]
    assert held_out == ["push"]


def test_the_boxs_footprint_is_5_8_cm_across_and_17_8_cm_along_table_y_at_angle_0():
    corners = tracking.object_corners([20.0, 30.0], 0.0, size=(5.8, 17.8, 3.0))
    np.testing.assert_allclose(corners.min(axis=0), [17.1, 21.1])
    np.testing.assert_allclose(corners.max(axis=0), [22.9, 38.9])
    # A quarter turn lays the long side along table x.
    turned = tracking.object_corners([20.0, 30.0], np.pi / 2, size=(5.8, 17.8, 3.0))
    np.testing.assert_allclose(np.ptp(turned, axis=0), [17.8, 5.8])


def test_a_clip_where_the_hand_is_found_less_than_a_quarter_second_before_the_object_moves_is_flagged():
    # The object is 1.25 cm from its start 0.625 s after the clap, and 0 cm one frame earlier. Found that late,
    # the palm path has almost nothing before contact (ticket 15 needs the approach).
    def late(clip):
        return any("hand found late" in flag for flag in clip.flags)

    assert not late(made_up_clip("early", 5.0, entered=0.25))  # 0.375 s before
    assert late(made_up_clip("late", 5.0, entered=0.5))  # 0.125 s before
    assert late(made_up_clip("after", 5.0, entered=0.75))  # only once it moves
    assert not late(made_up_clip("near-miss", 0.3, entered=0.75))  # the object never moves


def test_the_tracks_kept_next_to_the_positions_survive_saving_and_loading(tmp_path):
    episode = tracking.to_episodes([made_up_clip("a", 5.0)])[0]
    save_episodes([episode], tmp_path)
    loaded = load_episodes(tmp_path)[0]
    for name in ("pusher_seen", "object_seen", "object_angle", "hand_direction"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(episode, name))
    assert not loaded.pusher_seen[:2].any() and loaded.pusher_seen[2:].all()
