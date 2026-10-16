"""Seam A with a trained predictor: toy episodes + toy encoder -> results report."""

import dataclasses

import pytest
import torch

from pwm.datasets import toy
from pwm.encoders.toy import ToyEncoder
from pwm.evaluator import (
    BLIND,
    PREDICTOR,
    PROBE_FLOOR,
    REVERSED,
    SHUFFLED,
    STATIC,
    Report,
    evaluate,
    near_miss,
    winning_encoder,
)
from pwm.predictor import train

ENCODER = ToyEncoder()
TOY = dict(size="tiny", batch_size=32)
# Training to generalise takes ~1500 steps, about 25 s per predictor on a GPU and several
# minutes on a laptop CPU, so it uses the GPU when there is one, and those tests are slow.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NO_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
TRAINED = pytest.mark.slow  # the tests of the predictors trained to generalise


def embed(episodes):
    return [ENCODER.encode(ep.frames) for ep in episodes]


@pytest.fixture(scope="module")
def toy_episodes():
    return toy.make_episodes(n_episodes=60, seed=0)


@pytest.mark.parametrize(
    "device, precision",
    [pytest.param("cpu", "fp32", marks=pytest.mark.slow), pytest.param("cuda", "fp16", marks=NO_GPU)],  # CPU: 20 s
    ids=["cpu-fp32", "gpu-fp16"],
)
def test_predictor_memorises_one_episode(toy_episodes, device, precision):
    train_episodes = [ep for ep in toy_episodes if ep.split == "train"]
    one = next(ep for ep in train_episodes if ep.kind == "push")
    others = [ep for ep in train_episodes if ep is not one]

    predictor = train([one], embed([one]), toy.STEP, steps=300, device=device, precision=precision, **TOY)

    # Score it on the very episode it was trained on; the probe is fitted on the others.
    episodes = others + [dataclasses.replace(one, split="held-out")]
    report = evaluate(episodes, embed(episodes), toy.STEP, toy.MOVE_THRESHOLD, predictor=predictor.rollout)

    # Near-zero training error: the rollout is as good as the real future frames, at every horizon.
    errors, floor = report.errors[PREDICTOR]["all"], report.errors[PROBE_FLOOR]["all"]
    assert report.errors[STATIC]["all"][-1] > 2  # the object really moves in this episode
    assert all(e < f + 0.25 for e, f in zip(errors, floor)), (errors, floor)


@pytest.fixture(scope="module")
def toy_models(toy_episodes):
    """Predictor and action-blind predictor trained alike on the toy training episodes."""
    embeddings = embed(toy_episodes)
    predictor = train(toy_episodes, embeddings, toy.STEP, steps=1500, device=DEVICE, **TOY)
    blind = train(toy_episodes, embeddings, toy.STEP, use_actions=False, steps=1500, device=DEVICE, **TOY)
    return embeddings, predictor, blind


@pytest.fixture(scope="module")
def toy_report(toy_episodes, toy_models):
    embeddings, predictor, blind = toy_models
    return evaluate(
        toy_episodes, embeddings, toy.STEP, toy.MOVE_THRESHOLD, predictor=predictor.rollout, blind=blind.rollout
    )


def at_horizon_4_on_moving_windows(report, row):
    return report.errors[row]["moving"][report.horizons.index(4)]


@TRAINED
def test_trained_predictor_beats_both_baselines_on_held_out_episodes(toy_report):
    error = {row: at_horizon_4_on_moving_windows(toy_report, row) for row in (PREDICTOR, STATIC, BLIND)}
    assert error[PREDICTOR] < error[STATIC], error
    assert error[PREDICTOR] < error[BLIND], error


@TRAINED
def test_wrong_pusher_actions_raise_the_error(toy_report):
    error = {row: at_horizon_4_on_moving_windows(toy_report, row) for row in (PREDICTOR, REVERSED, SHUFFLED)}
    assert error[REVERSED] > error[PREDICTOR], error
    assert error[SHUFFLED] > error[PREDICTOR], error


@TRAINED
def test_the_predictor_keeps_the_object_still_when_the_pusher_misses_it(toy_episodes, toy_models):
    embeddings, predictor, blind = toy_models
    test = near_miss(toy_episodes, embeddings, toy.STEP, toy.MOVE_THRESHOLD, predictor=predictor.rollout,
                     blind=blind.rollout)
    h = test.horizons.index(4)
    missed, pushed = test.displacement["near-miss"], test.displacement["push"]
    # Near-miss windows: the pusher moves past the object, which stays where it is.
    assert test.n_windows["near-miss"][h] > 0 and missed["true"][h] == 0
    # Push windows: the object moves; the predictor must see it move.
    assert pushed["true"][h] > 2
    # The predictor moves the object a little on near-misses, and much more on pushes.
    assert missed[PREDICTOR][h] < 1.0, missed
    assert missed[PREDICTOR][h] < pushed[PREDICTOR][h] / 3, (missed, pushed)


NAMES = {"predictor": PREDICTOR, "blind": BLIND, "reversed": REVERSED, "shuffled": SHUFFLED}


def hand_made_report(probe_error=1.0, workspace=50.0, **at_horizon_4):
    """A report whose rows have the given error at horizon 4 on moving windows (others: 99)."""
    errors = {row: {"moving": [99, 99, e, 99], "all": [99] * 4} for row, e in at_horizon_4.items()}
    return Report(
        unit="cm",
        workspace=workspace,
        step=0.25,
        move_threshold=1.0,
        probe_error=probe_error,
        horizons=[1, 2, 4, 8],
        n_windows=100,
        n_moving=[10, 20, 30, 40],
        errors={STATIC: errors.pop("static"), **{NAMES[row]: err for row, err in errors.items()}},
    )


def test_gate_1_passes_when_the_probe_error_is_under_4_percent_of_the_workspace_width():
    # 4% of a 50 cm workspace is 2 cm.
    assert hand_made_report(probe_error=1.99, static=1).gate_1
    assert not hand_made_report(probe_error=2.01, static=1).gate_1


@pytest.mark.parametrize(
    "errors, passes",
    [
        # The better baseline is the action-blind predictor (10): the pass mark is 8.
        (dict(static=12, blind=10, predictor=7.99, reversed=9, shuffled=9), True),
        (dict(static=12, blind=10, predictor=8.01, reversed=9, shuffled=9), False),
        # The better baseline is the static baseline (10): the pass mark is 8.
        (dict(static=10, blind=12, predictor=7.99, reversed=9, shuffled=9), True),
        (dict(static=10, blind=12, predictor=8.01, reversed=9, shuffled=9), False),
        # Both wrong-action rows must raise the error.
        (dict(static=12, blind=10, predictor=7, reversed=7, shuffled=9), False),
        (dict(static=12, blind=10, predictor=7, reversed=9, shuffled=6), False),
    ],
)
def test_validation_gate_follows_the_fixed_pass_mark(errors, passes):
    assert hand_made_report(**errors).validation_gate is passes


def test_validation_gate_is_not_reported_without_a_predictor():
    assert hand_made_report(static=10).validation_gate is None


def test_the_winning_encoder_has_the_lower_predictor_error_at_horizon_4_on_moving_windows():
    reports = {
        "dinov2": hand_made_report(static=40, blind=30, predictor=20, reversed=25, shuffled=25),
        "vjepa2": hand_made_report(static=50, blind=45, predictor=15, reversed=25, shuffled=25),
    }
    assert winning_encoder(reports) == "vjepa2"
