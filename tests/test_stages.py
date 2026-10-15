"""Seam A through the stage scripts: each stage reads and writes the run folder."""

import dataclasses
import json

import numpy as np

from pwm import stages
from pwm.encoders import ENCODERS
from pwm.encoders.toy import ToyEncoder
from pwm.episodes import load_episodes, save_episodes


def test_stages_write_the_static_baseline_report_to_the_run_folder(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    stages.evaluate("toy", "toy", root=tmp_path)

    results_dir = tmp_path / "toy" / "toy"
    results = json.loads((results_dir / "results.json").read_text())
    static = results["errors"]["static baseline"]
    assert results["horizons"] == [1, 2, 4, 8]
    # Still windows flatter the static baseline: it looks better overall than on moving windows.
    assert all(moving > overall for moving, overall in zip(static["moving"], static["all"]))
    assert "static baseline" in (results_dir / "report.md").read_text()


def test_stages_report_the_trained_predictors_and_the_gates(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3)
    stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3)
    stages.evaluate("toy", "toy", root=tmp_path)

    results = json.loads((tmp_path / "toy" / "toy" / "results.json").read_text())
    rows = {"predictor", "action-blind predictor", "predictor, reversed actions", "predictor, shuffled actions"}
    assert rows <= set(results["errors"])
    assert results["gate_1"] is True  # the toy probe is near perfect
    assert results["validation_gate"] in (True, False)


def test_embedding_again_encodes_only_the_episodes_not_cached_yet(tmp_path, monkeypatch):
    frames_seen = []

    class CountingEncoder(ToyEncoder):
        name = "counting"

        def encode(self, frames):
            frames_seen.append(len(frames))
            return super().encode(frames)

    monkeypatch.setitem(ENCODERS, "counting", CountingEncoder)
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "counting", root=tmp_path)
    episodes_dir = tmp_path / "toy" / "episodes"
    new = dataclasses.replace(load_episodes(episodes_dir)[0], name="toy-new")
    save_episodes([new], episodes_dir)
    frames_seen.clear()

    stages.embed("toy", "counting", root=tmp_path)

    assert frames_seen == [40]  # only the new episode, 40 frames
    episodes = load_episodes(episodes_dir)
    embeddings = stages.load_embeddings("toy", "counting", episodes, root=tmp_path)
    assert len(embeddings) == 61
    np.testing.assert_allclose(embeddings[-1], ToyEncoder().encode(new.frames), atol=1e-3)
