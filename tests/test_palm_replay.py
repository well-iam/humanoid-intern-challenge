"""My palm paths -> pushes for the Panda replay: the contact shift and where a replay starts, on made-up
episodes (fast). The replay itself is Seam C in test_panda_replay.py (slow).
"""

import numpy as np

from pwm.episodes import Episode
from pwm.replay import palm

BOX = (5.8, 17.8, 3.0)


def made_up_push(name="push", reach=12.0, found=0.25, direction=(0.0, -1.0), n=24, moved=True) -> Episode:
    """The hand comes up the table (-y) at 8 cm/s (1 cm a frame), touching with fingertips `reach` cm ahead of the
    palm, and pushes a box that stands upright (angle 0) at (25, 25), its near edge at y = 33.9. The palm is at
    (25, 55 - 8 t) and the box starts moving once the fingertips pass that edge; the hand is found `found` s
    after the clap, and is held at that point before."""
    t = np.arange(n) / 8
    palm_y = 55.0 - 8.0 * t
    box_y = 25.0 - np.clip(33.9 - (palm_y - reach), 0, None) if moved else np.full(n, 25.0)
    seen = t >= found
    pusher = np.stack([np.full(n, 25.0), palm_y], axis=1)
    pusher[~seen] = pusher[np.argmax(seen)]
    return Episode(
        name=name,
        frames=np.zeros((n, 8, 8, 3), np.uint8),
        timestamps=t,
        pusher_pos=pusher,
        object_pos=np.stack([np.full(n, 25.0), box_y], axis=1),
        workspace=50.0,
        unit="cm",
        split="train",
        kind="push" if moved else "near-miss",
        pusher_seen=seen,
        object_seen=np.ones(n, bool),
        object_angle=np.zeros(n),
        hand_direction=np.tile(direction, (n, 1)),
    )


def test_the_shift_runs_from_the_palm_to_the_nearest_point_of_the_box_one_frame_before_it_moves():
    # The box first moves 0.9 cm at frame 10 (palm at y = 45); one frame earlier the palm is at (25, 46), and the
    # nearest point of the box's edge is (25, 33.9).
    shift = palm.own_shift(made_up_push(), size=BOX)
    np.testing.assert_allclose(shift, [0.0, -12.1], atol=1e-9)


def test_the_replay_starts_where_the_hand_is_first_found_and_moves_the_whole_path_by_one_vector():
    # Before the hand is found the palm is only held at its first tracked point, so the replay starts there, with
    # the box where it was then. Every palm position after that moves by the same shift: no jump at contact.
    ep = made_up_push(found=0.5)
    shift = np.array([1.0, -12.1])

    push = palm.clip_push(ep, shift, size=BOX, gripper=(0.0, 0.0))

    np.testing.assert_allclose(push.timestamps, ep.timestamps[4:])
    np.testing.assert_allclose(push.path, ep.pusher_pos[4:] + shift)
    np.testing.assert_allclose(push.start, [25.0, 25.0, 0.0])
    assert (push.shape, push.unit, push.workspace, push.size) == ("box", "cm", 50.0, BOX)


def test_clips_without_a_palm_at_contact_get_the_median_shift_turned_to_their_own_hand_direction():
    # Two pushes with the hand found in time: shifts 12.1 and 14.1 cm straight ahead of the hand. A push whose
    # hand is found only after the box moved, and a near-miss whose hand points along +x, get the median,
    # 13.1 cm ahead of their own hand. The hand comes in from every side, so the median lives in the hand's frame.
    episodes = [
        made_up_push("a", reach=12.0),
        made_up_push("b", reach=14.0),
        made_up_push("late", found=1.5),
        made_up_push("near-miss", direction=(1.0, 0.0), moved=False),
    ]

    shifts = palm.shifts(episodes, size=BOX)

    np.testing.assert_allclose(shifts["a"].vector, [0.0, -12.1], atol=1e-9)
    np.testing.assert_allclose(shifts["b"].vector, [0.0, -14.1], atol=1e-9)
    np.testing.assert_allclose(shifts["late"].vector, [0.0, -13.1], atol=1e-9)
    np.testing.assert_allclose(shifts["near-miss"].vector, [13.1, 0.0], atol=1e-9)
    assert [shifts[name].own for name in ("a", "b", "late", "near-miss")] == [True, True, False, False]


def test_a_hand_found_less_than_a_quarter_second_before_the_box_moves_gets_no_shift_of_its_own():
    # The box first moves at 1.25 s. The palm at contact is known in both, but a hand found 0.125 s before
    # contact has only just been found: its first positions are not trusted (the tracking report flags it).
    in_time = palm.shifts([made_up_push("in time", found=1.0), made_up_push("other")], size=BOX)
    late = palm.shifts([made_up_push("late", found=1.125), made_up_push("other")], size=BOX)
    assert in_time["in time"].own and not late["late"].own


def test_the_gripper_stops_short_of_the_edge_by_its_own_half_size_along_the_shift():
    # The Panda's closed fingers reach 1.9 cm along table y and 1 cm along table x from their centre.
    ep = made_up_push()
    up = palm.clip_push(ep, np.array([0.0, -12.1]), size=BOX, gripper=(1.0, 1.9))
    right = palm.clip_push(ep, np.array([12.1, 0.0]), size=BOX, gripper=(1.0, 1.9))
    np.testing.assert_allclose(up.path - ep.pusher_pos[2:], np.tile([0.0, -10.2], (len(up.path), 1)))
    np.testing.assert_allclose(right.path - ep.pusher_pos[2:], np.tile([11.1, 0.0], (len(right.path), 1)))
