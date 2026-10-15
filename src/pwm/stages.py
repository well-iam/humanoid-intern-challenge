"""The pipeline stages and the run folder they share.

Each stage takes a dataset (and an encoder), reads its inputs from the run folder and
writes its outputs there, so any stage can be rerun on its own:

    <root>/<dataset>/raw/                    raw source, owned by the dataset adapter
    <root>/<dataset>/episodes/<name>.npz     one file per episode         (make_episodes)
    <root>/<dataset>/<encoder>/embeddings/<name>.npy  fp16 (frames, patches, dims)  (embed)
    <root>/<dataset>/<encoder>/predictor.pt    the trained predictor                  (train)
    <root>/<dataset>/<encoder>/predictor-blind.pt  the action-blind predictor        (train --blind)
    <root>/<dataset>/<encoder>/results.json    the report, machine-readable           (evaluate)
    <root>/<dataset>/<encoder>/report.md       the report, for people                 (evaluate)

The root is the `root` argument, else the PWM_DATA environment variable, else ./data.
"""

import os
from pathlib import Path

import numpy as np
import torch

from pwm.datasets import DATASETS
from pwm.encoders import get_encoder
from pwm.episodes import Episode, load_episodes, save_episodes
from pwm.evaluator import Report
from pwm.evaluator import evaluate as evaluate_episodes
from pwm.predictor import Predictor
from pwm.predictor import train as train_predictor


def data_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("PWM_DATA") or "data")


def make_episodes(dataset: str, root: str | Path | None = None) -> None:
    folder = data_root(root) / dataset
    episodes = DATASETS[dataset].build(folder / "raw")
    save_episodes(episodes, folder / "episodes")


def embed(dataset: str, encoder: str, root: str | Path | None = None) -> None:
    """Encode every episode that has no cached frame embeddings yet (or a different frame count)."""
    folder = data_root(root) / dataset
    out = folder / encoder / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    enc = None
    for ep in load_episodes(folder / "episodes"):
        path = out / f"{ep.name}.npy"
        if path.exists() and len(np.load(path, mmap_mode="r")) == len(ep.frames):
            continue
        enc = enc or get_encoder(encoder)  # load the model only if there is work to do
        partial = out / f"{ep.name}.partial.npy"
        np.save(partial, enc.encode(ep.frames).astype(np.float16))
        partial.replace(path)  # an interrupted run never leaves a half-written cache file


def load_embeddings(dataset: str, encoder: str, episodes: list[Episode], root=None) -> list[np.ndarray]:
    """Memory-mapped frame embeddings of each episode, in the order of `episodes`."""
    folder = data_root(root) / dataset / encoder / "embeddings"
    views = []
    for ep in episodes:
        view = np.load(folder / f"{ep.name}.npy", mmap_mode="r")
        assert len(view) == len(ep.frames), ep.name
        views.append(view)
    return views


def train(
    dataset: str,
    encoder: str,
    root: str | Path | None = None,
    blind: bool = False,
    device: str | None = None,
    **options,
) -> None:
    """Train the predictor (or, with `blind`, the action-blind predictor) on the training episodes.

    `options` go to `pwm.predictor.train` (size, steps, batch_size, lr, precision, ...).
    """
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    predictor = train_predictor(
        episodes, embeddings, DATASETS[dataset].step, use_actions=not blind, device=device or _device(), **options
    )
    predictor.save(folder / encoder / _predictor_file(blind))


def evaluate(dataset: str, encoder: str, root: str | Path | None = None) -> Report:
    """Score the static baseline and whichever predictors have been trained."""
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    spec = DATASETS[dataset]
    trained = {}
    for role, blind in (("predictor", False), ("blind", True)):
        path = folder / encoder / _predictor_file(blind)
        if path.exists():
            trained[role] = Predictor.load(path, _device()).rollout
    report = evaluate_episodes(episodes, embeddings, spec.step, spec.move_threshold, **trained)
    report.save(folder / encoder, title=f"{dataset}, {encoder} encoder")
    return report


def _predictor_file(blind: bool) -> str:
    return "predictor-blind.pt" if blind else "predictor.pt"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
