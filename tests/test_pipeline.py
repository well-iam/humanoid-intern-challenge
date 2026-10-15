"""Seam A: episodes + encoder -> results report, on the toy world with the toy encoder."""

import dataclasses

import numpy as np
import pytest

from pwm.datasets import toy
from pwm.encoders.toy import ToyEncoder
from pwm.episodes import Episode
from pwm.evaluator import PROBE_FLOOR, STATIC, evaluate


def run(episodes):
    encoder = ToyEncoder()
    embeddings = [encoder.encode(ep.frames) for ep in episodes]
    return evaluate(episodes, embeddings, step=toy.STEP, move_threshold=toy.MOVE_THRESHOLD)


@pytest.fixture(scope="module")
def toy_episodes():
    return toy.make_episodes(n_episodes=60, seed=0)


def train_only(episodes):
    return [ep for ep in episodes if ep.split == "train"]


def still_episode(name, n_frames=40, fps=10):
    """A held-out episode in which nothing moves: blank frames, pusher and object at rest."""
    return Episode(
        name=name,
        frames=np.zeros((n_frames, toy.IMAGE_SIZE, toy.IMAGE_SIZE, 3), np.uint8),
        timestamps=np.arange(n_frames) / fps,
        pusher_pos=np.tile([10.0, 10.0], (n_frames, 1)),
        object_pos=np.tile([32.0, 32.0], (n_frames, 1)),
        workspace=toy.WORKSPACE,
        unit=toy.UNIT,
        split="held-out",
        kind="near-miss",
    )


def frozen_at_first_frame(ep):
    n = len(ep.frames)
    return dataclasses.replace(
        ep,
        frames=np.repeat(ep.frames[:1], n, axis=0),
        pusher_pos=np.repeat(ep.pusher_pos[:1], n, axis=0),
        object_pos=np.repeat(ep.object_pos[:1], n, axis=0),
    )


def test_probe_recovers_object_position_on_held_out_episodes(toy_episodes):
    report = run(toy_episodes)
    # Gate 1 pass mark: under 4% of the workspace width.
    assert report.probe_error < 0.04 * toy.WORKSPACE


def test_a_loud_channel_that_carries_nothing_does_not_hide_the_object_from_the_probe(toy_episodes):
    # Real ViTs have a few channels with huge activations. Here one extra channel, the same
    # in every patch, holds noise 100x louder than the colours that show the object.
    rng = np.random.default_rng(0)
    encoder = ToyEncoder()
    embeddings = []
    for ep in toy_episodes:
        e = encoder.encode(ep.frames)
        loud = 100 * rng.standard_normal((len(e), 1, 1)) * np.ones((1, e.shape[1], 1))
        embeddings.append(np.concatenate([e, loud.astype(np.float32)], axis=-1))

    report = evaluate(toy_episodes, embeddings, step=toy.STEP, move_threshold=toy.MOVE_THRESHOLD)

    assert report.probe_error < 0.04 * toy.WORKSPACE


def test_static_baseline_error_is_zero_where_nothing_moves(toy_episodes):
    # Freeze each held-out toy episode at its first frame, so nothing moves in any window.
    frozen = [frozen_at_first_frame(ep) if ep.split == "held-out" else ep for ep in toy_episodes]

    report = run(frozen)

    assert report.n_moving == [0, 0, 0, 0]
    # The error is measured against the true object position, so what is left is the
    # probe's own error on the real frame: the static baseline adds nothing to it.
    assert report.errors[STATIC]["all"] == report.errors[PROBE_FLOOR]["all"]


def test_every_starting_offset_gives_a_window(toy_episodes):
    # 40 frames at 10 Hz, step 0.2 s = every 2nd frame. A window spans 3 context frames
    # and 8 future steps, i.e. 10 steps = 20 frames, so it can start at frames 0..19.
    report = run(train_only(toy_episodes) + [still_episode("still")])
    assert report.n_windows == 20


def test_moving_windows_are_those_where_the_object_moves_more_than_the_threshold_within_the_horizon(
    toy_episodes,
):
    jump = still_episode("jump")
    jump.object_pos[16:] += [2.0, 0.0]  # above the 1-unit threshold, between frames 15 and 16
    nudge = still_episode("nudge")
    nudge.object_pos[16:] += [0.5, 0.0]  # below the threshold: never a moving window

    report = run(train_only(toy_episodes) + [jump, nudge])

    # The current frame (last context frame) of the window starting at frame s is s + 4,
    # so s + 4 is in 4..23. A window moves at horizon h if the current frame is <= 15
    # and the frame h steps ahead is >= 16: current frames 14-15, 12-15, 8-15, 4-15.
    assert report.n_moving == [2, 4, 8, 12]
