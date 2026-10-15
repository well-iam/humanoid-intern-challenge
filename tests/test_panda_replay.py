"""Seam C for the Panda replay: pusher path + object start pose -> final object pose.

Slow: runs MuJoCo, and the first run downloads the Menagerie Panda (about 35 MB) into
$PWM_DATA/models (else ./data/models).
"""

import numpy as np
import pytest

from pwm.replay import Push, get_replay

pytestmark = pytest.mark.slow


def square_path(centre=25.0, side=20.0, speed=10.0, rate=10.0):
    """A square pusher path in cm, run at `speed` cm/s and sampled at `rate` Hz."""
    corners = centre + side / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(corners, axis=0), axis=1))])
    timestamps = np.arange(0, length[-1] / speed + 1e-9, 1 / rate)
    path = np.stack([np.interp(timestamps * speed, length, corners[:, i]) for i in (0, 1)], axis=1)
    return path, timestamps


@pytest.mark.parametrize("floating", [False, True], ids=["arm", "floating gripper"])
def test_the_gripper_follows_a_square_path_within_a_centimetre(floating):
    path, timestamps = square_path()
    box_far_away = np.array([45.0, 5.0, 0.0])
    push = Push(path=path, timestamps=timestamps, start=box_far_away, shape="box", unit="cm",
                workspace=50.0, size=(6.0, 8.0, 3.0))

    result = get_replay("panda", floating=floating).replay(push)

    error = np.linalg.norm(result.pusher - path, axis=1)
    assert error.max() < 1.0, f"worst tracking error {error.max():.2f} cm at step {error.argmax()}"


@pytest.mark.parametrize("floating", [False, True], ids=["arm", "floating gripper"])
def test_a_straight_push_moves_the_box_in_the_push_direction(floating):
    # The pusher's centre starts 13 cm below the box's near edge and moves 20 cm up the table
    # (-y), so the box should travel about 7 cm (less the fingers' half-width) up and hardly sideways.
    timestamps = np.arange(0, 4.0 + 1e-9, 0.1)
    path = np.stack([np.full_like(timestamps, 25.0), 42.0 - 5.0 * timestamps], axis=1)
    start = np.array([25.0, 25.0, 0.0])
    push = Push(path=path, timestamps=timestamps, start=start, shape="box", unit="cm",
                workspace=50.0, size=(6.0, 8.0, 3.0))

    result = get_replay("panda", floating=floating).replay(push)

    dx, dy = result.final[:2] - start[:2]
    assert -11.0 < dy < -5.0, dy
    assert abs(dx) < 0.25 * abs(dy), dx
    assert np.linalg.norm(result.pusher - path, axis=1).max() < 1.0  # it still follows while pushing


def test_an_untouched_t_stays_put_and_is_drawn_where_push_t_draws_it():
    # Same start as the Push-T judge's test: the pusher runs along the top wall, far from the T.
    start = np.array([300.0, 250.0, 0.7])
    path = np.stack([np.linspace(40, 470, 21), np.full(21, 40.0)], axis=1)
    push = Push(path=path, timestamps=np.arange(21) * 0.1, start=start, shape="T", unit="Push-T units")

    panda = get_replay("panda").replay(push, frame_size=96)
    judge = get_replay("pusht").replay(push, frame_size=96)

    np.testing.assert_allclose(panda.final, start, atol=0.5)
    assert panda.frames.shape == (21, 96, 96, 3) and panda.frames.dtype == np.uint8
    # The T is slate grey in both: bluer than the table and the arm, much less blue than Push-T's pusher.
    # In the first frame the arm reaches to the top-left corner, clear of the T.
    t_mask = lambda frame: (lambda blue: (blue > 15) & (blue < 60))(frame[..., 2].astype(int) - frame[..., 0])
    a, b = t_mask(panda.frames[0]), t_mask(judge.frames[0])
    assert (a & b).sum() / (a | b).sum() > 0.7
