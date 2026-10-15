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


def test_a_run_trained_with_another_seed_lives_in_its_own_folder_and_leaves_the_default_run_alone(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3)
    stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3)
    stages.evaluate("toy", "toy", root=tmp_path)
    default = tmp_path / "toy" / "toy"
    before = {name: (default / name).read_bytes() for name in ("predictor.pt", "predictor-blind.pt", "results.json")}

    stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3, seed=1)
    stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3, seed=1)
    stages.evaluate("toy", "toy", root=tmp_path, seed=1)

    assert {name: (default / name).read_bytes() for name in before} == before
    seeded = json.loads((default / "seed-1" / "results.json").read_text())
    ours = json.loads(before["results.json"])
    assert seeded["validation_gate"] in (True, False)
    assert seeded["errors"]["predictor"] != ours["errors"]["predictor"]  # another model
    assert seeded["errors"]["static baseline"] == ours["errors"]["static baseline"]  # same held-out windows
    assert stages.compare("toy", root=tmp_path) == "toy"  # a seed run is not mistaken for an encoder


def test_comparing_encoders_picks_the_one_that_predicts_the_object_better_and_plots_both(tmp_path, monkeypatch):
    class BlurryEncoder(ToyEncoder):
        """The toy encoder with noise on top: it shows the object less clearly."""

        name = "blurry"

        def encode(self, frames):
            clean = super().encode(frames)
            return clean + 0.2 * np.random.default_rng(len(frames)).standard_normal(clean.shape).astype(np.float32)

    monkeypatch.setitem(ENCODERS, "blurry", BlurryEncoder)
    stages.make_episodes("toy", root=tmp_path)
    for encoder in ("toy", "blurry"):
        stages.embed("toy", encoder, root=tmp_path)
        stages.train("toy", encoder, root=tmp_path, size="tiny", steps=3)
        stages.train("toy", encoder, root=tmp_path, blind=True, size="tiny", steps=3)
        stages.evaluate("toy", encoder, root=tmp_path)

    winner = stages.compare("toy", root=tmp_path)

    assert winner == "toy"
    assert json.loads((tmp_path / "toy" / "winner.json").read_text())["encoder"] == "toy"
    assert (tmp_path / "toy" / "error-vs-horizon.png").read_bytes().startswith(b"\x89PNG")


def test_retrievals_draw_the_nearest_real_frames_of_a_few_held_out_rollouts(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3)
    stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3)

    path = stages.retrievals("toy", "toy", root=tmp_path, n_rollouts=2)

    assert path == tmp_path / "toy" / "toy" / "retrievals.png"
    assert path.read_bytes().startswith(b"\x89PNG")


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
