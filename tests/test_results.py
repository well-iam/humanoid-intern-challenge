"""Seam A, the visuals of the results report: nearest-real-frame retrievals and plots."""

import numpy as np
import pytest

from pwm.datasets import toy
from pwm.encoders.toy import ToyEncoder
from pwm.evaluator import nearest_frames

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
