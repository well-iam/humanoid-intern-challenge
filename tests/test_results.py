"""Seam A, the visuals of the results report: nearest-real-frame retrievals and plots."""

import numpy as np
import pytest

from pwm.datasets import toy
from pwm.encoders.toy import ToyEncoder
from pwm.evaluator import BLIND, PREDICTOR, REVERSED, SHUFFLED, STATIC, Report, nearest_frames, seeds_to_markdown

ENCODER = ToyEncoder()


@pytest.fixture(scope="module")
def toy_episodes():
    return toy.make_episodes(n_episodes=24, seed=0)


def test_the_nearest_real_frame_to_a_frame_embedding_shows_the_object_where_that_frame_does(toy_episodes):
    bank = [ep for ep in toy_episodes if ep.split == "train"]
    bank_embeddings = [ENCODER.encode(ep.frames) for ep in bank]
    query_ep, query_frames = bank[3], [5, 20, 39]
    queries = bank_embeddings[3][query_frames]
    queries = queries + 1e-3 * np.random.default_rng(0).standard_normal(queries.shape)  # not bit-identical

    found = nearest_frames(queries, bank_embeddings)

    for (e, f), frame in zip(found, query_frames):
        np.testing.assert_allclose(bank[e].object_pos[f], query_ep.object_pos[frame], atol=0.5)


def test_the_seed_summary_gives_each_seeds_verdict_and_the_mean_and_spread_at_the_gate():
    def report(predictor, blind):
        rows = {STATIC: 40.0, PREDICTOR: predictor, BLIND: blind, REVERSED: 45.0, SHUFFLED: 38.0}
        return Report(unit="units", workspace=512, step=0.5, move_threshold=2, probe_error=4, horizons=[4],
                      n_windows=10, n_moving=[8], errors={k: {"moving": [v], "all": [v]} for k, v in rows.items()})

    # Pass mark = 0.8 x the action-blind error 24 = 19.2: seed 0 passes by 1.2, the others miss.
    text = seeds_to_markdown({0: report(18.0, 24.0), 1: report(20.0, 24.0), 2: report(22.0, 24.0)}, "toy")

    lines = text.splitlines()
    assert "| seed 0 | 18.00 | 24.00 | 40.00 | 45.00 | 38.00 | 19.20 | 1.20 | pass |" in lines
    assert "| seed 2 | 22.00 | 24.00 | 40.00 | 45.00 | 38.00 | 19.20 | -2.80 | fail |" in lines
    assert ("| mean ± std | 20.00 ± 2.00 | 24.00 ± 0.00 | 40.00 ± 0.00 | 45.00 ± 0.00 | 38.00 ± 0.00 "
            "| 19.20 ± 0.00 | -0.80 ± 2.00 | 1 of 3 |") in lines
    assert "| static baseline, every seed | 40.00 |" in lines  # the same for every seed: given once
