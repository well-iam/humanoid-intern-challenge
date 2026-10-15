"""The pipeline stages and the run folder they share.

Each stage takes a dataset (and an encoder), reads its inputs from the run folder and
writes its outputs there, so any stage can be rerun on its own:

    <root>/<dataset>/raw/                    raw source, owned by the dataset adapter
    <root>/<dataset>/episodes/<name>.npz     one file per episode         (make_episodes)
    <root>/<dataset>/<encoder>/embeddings.npy  fp16, all frames in episode-name order  (embed)
    <root>/<dataset>/<encoder>/embeddings.json episode names and frame counts           (embed)
    <root>/<dataset>/<encoder>/results.json    the report, machine-readable           (evaluate)
    <root>/<dataset>/<encoder>/report.md       the report, for people                 (evaluate)

The root is the `root` argument, else the PWM_DATA environment variable, else ./data.
"""

import json
import os
from pathlib import Path

import numpy as np

from pwm.datasets import DATASETS
from pwm.encoders import get_encoder
from pwm.episodes import Episode, load_episodes, save_episodes
from pwm.evaluator import Report
from pwm.evaluator import evaluate as evaluate_episodes


def data_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("PWM_DATA") or "data")


def make_episodes(dataset: str, root: str | Path | None = None) -> None:
    folder = data_root(root) / dataset
    episodes = DATASETS[dataset].build(folder / "raw")
    save_episodes(episodes, folder / "episodes")


def embed(dataset: str, encoder: str, root: str | Path | None = None) -> None:
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    enc = get_encoder(encoder)
    out = folder / encoder
    out.mkdir(parents=True, exist_ok=True)

    total = sum(len(ep.frames) for ep in episodes)
    store, offset = None, 0
    for ep in episodes:
        e = enc.encode(ep.frames)
        if store is None:
            store = np.lib.format.open_memmap(out / "embeddings.npy", "w+", np.float16, (total, *e.shape[1:]))
        store[offset : offset + len(e)] = e
        offset += len(e)
    store.flush()
    (out / "embeddings.json").write_text(json.dumps({"episodes": [[ep.name, len(ep.frames)] for ep in episodes]}))


def load_embeddings(dataset: str, encoder: str, episodes: list[Episode], root=None) -> list[np.ndarray]:
    """Memory-mapped frame embeddings of each episode, in the order of `episodes`."""
    folder = data_root(root) / dataset / encoder
    store = np.load(folder / "embeddings.npy", mmap_mode="r")
    index = json.loads((folder / "embeddings.json").read_text())["episodes"]
    starts, offset = {}, 0
    for name, n in index:
        starts[name] = (offset, n)
        offset += n
    views = []
    for ep in episodes:
        start, n = starts[ep.name]
        assert n == len(ep.frames), ep.name
        views.append(store[start : start + n])
    return views


def evaluate(dataset: str, encoder: str, root: str | Path | None = None) -> Report:
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    spec = DATASETS[dataset]
    report = evaluate_episodes(episodes, embeddings, spec.step, spec.move_threshold)
    report.save(folder / encoder, title=f"{dataset}, {encoder} encoder")
    return report
