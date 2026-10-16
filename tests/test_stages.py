"""Seam A through the stage scripts: each stage reads and writes the run folder."""

import dataclasses
import json

import numpy as np

from pwm import stages
from pwm.datasets import DATASETS, Dataset, toy
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
    assert (tmp_path / "toy" / "toy" / "error-vs-horizon.png").stat().st_size > 0


def test_the_near_miss_test_writes_its_table_and_plot_to_the_run_folder(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3)
    stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3)

    test = stages.near_miss("toy", "toy", root=tmp_path)

    folder = tmp_path / "toy" / "toy"
    saved = json.loads((folder / "near-miss.json").read_text())
    assert saved["displacement"]["near-miss"]["true"] == test.displacement["near-miss"]["true"]
    assert {"predictor", "action-blind predictor"} <= set(saved["displacement"]["push"])
    assert "near-miss windows" in (folder / "near-miss.md").read_text()
    assert (folder / "near-miss.png").stat().st_size > 0


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


def test_the_seed_summary_gives_each_seeds_gate_verdict_and_plots_each_seed(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    for seed in (0, 1):
        stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3, seed=seed)
        stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3, seed=seed)
        stages.evaluate("toy", "toy", root=tmp_path, seed=seed)

    summary = stages.seeds("toy", "toy", [0, 1], root=tmp_path)

    folder = tmp_path / "toy" / "toy"
    assert summary == folder / "seeds.md"
    lines = summary.read_text().splitlines()
    for seed in (0, 1):
        passed = json.loads((stages.run_folder("toy", "toy", tmp_path, seed) / "results.json").read_text())["validation_gate"]
        gate_line = next(line for line in lines if line.startswith(f"| seed {seed} |"))
        assert gate_line.rstrip(" |").endswith("pass" if passed else "fail")
    assert (folder / "error-vs-horizon-seeds.png").read_bytes().startswith(b"\x89PNG")


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


def test_rebuilding_the_episodes_drops_those_whose_source_is_gone(tmp_path):
    # E.g. a bad take deleted from the raw folder: its clips must not stay in the dataset.
    stages.make_episodes("toy", root=tmp_path)
    folder = tmp_path / "toy" / "episodes"
    save_episodes([dataclasses.replace(load_episodes(folder)[0], name="toy-gone")], folder)

    stages.make_episodes("toy", root=tmp_path)

    names = [ep.name for ep in load_episodes(folder)]
    assert "toy-gone" not in names and len(names) == 60


def test_a_dataset_can_read_its_raw_source_from_another_folder(tmp_path, monkeypatch):
    # The test take lives in <root>/clips/test-take but is a dataset of its own, never part of "clips".
    read_from = []

    def build(raw_dir):
        read_from.append(raw_dir)
        return toy.make_episodes(n_episodes=4)

    monkeypatch.setitem(DATASETS, "elsewhere", Dataset(build=build, step=0.2, move_threshold=1.0, raw="clips/test-take"))
    stages.make_episodes("elsewhere", root=tmp_path)

    assert read_from == [tmp_path / "clips" / "test-take"]
    assert len(load_episodes(tmp_path / "elsewhere" / "episodes")) == 4


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


def test_the_figures_are_redrawn_from_the_saved_results_alone(tmp_path):
    # The notebook's fast path: no episodes, no embeddings, no predictors, only the saved results.
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    for seed in (0, 1):
        stages.train("toy", "toy", root=tmp_path, size="tiny", steps=3, seed=seed)
        stages.train("toy", "toy", root=tmp_path, blind=True, size="tiny", steps=3, seed=seed)
        stages.evaluate("toy", "toy", root=tmp_path, seed=seed)
    stages.near_miss("toy", "toy", root=tmp_path)
    for path in [*tmp_path.rglob("*.png"), *tmp_path.rglob("*.npz"), *tmp_path.rglob("*.npy"), *tmp_path.rglob("*.pt")]:
        path.unlink()

    drawn = stages.redraw(root=tmp_path)

    folder = tmp_path / "toy" / "toy"
    expected = {folder / "error-vs-horizon.png", folder / "seed-1" / "error-vs-horizon.png",
                folder / "error-vs-horizon-seeds.png", folder / "near-miss.png"}
    assert set(drawn) == expected
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in expected)
