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
    <root>/<dataset>/<encoder>/retrievals.png  nearest real frames of a few rollouts  (retrievals)
    <root>/<dataset>/<encoder>/seed-<n>/       predictor.pt, predictor-blind.pt, results.json and
                                               report.md of a run trained with seed n (train and
                                               evaluate with seed=n; seed 0 is the encoder folder)
    <root>/<dataset>/<encoder>/seeds.md        the runs of several seeds side by side  (seeds)
    <root>/<dataset>/<encoder>/error-vs-horizon-seeds.png  one row per seed           (seeds)
    <root>/<dataset>/error-vs-horizon.png      every evaluated encoder, one row each  (compare)
    <root>/<dataset>/winner.json               the winning encoder and why            (compare)
    <root>/<dataset>/<encoder>/ranking/        ranking.json, report.md, ranking.png   (rank)

The root is the `root` argument, else the PWM_DATA environment variable, else ./data.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from pwm.datasets import DATASETS
from pwm.encoders import get_encoder
from pwm.episodes import Episode, load_episodes, save_episodes
from pwm.evaluator import (
    GATE_HORIZON,
    HORIZONS,
    PREDICTOR,
    Report,
    nearest_frames,
    seeds_to_markdown,
    winning_encoder,
)
from pwm.evaluator import evaluate as evaluate_episodes
from pwm.plots import ENCODER_NAMES, plot_errors, plot_ranking, save_retrievals
from pwm.predictor import Predictor
from pwm.predictor import train as train_predictor
from pwm.probe import Probe
from pwm.ranking import RankingReport, candidate_paths
from pwm.windows import CONTEXT, make_windows


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
    seed: int = 0,
    **options,
) -> None:
    """Train the predictor (or, with `blind`, the action-blind predictor) on the training episodes.

    `seed` sets the initial weights and the batches; a seed other than 0 is a separate run,
    saved in its own folder. `options` go to `pwm.predictor.train` (size, steps, batch_size,
    lr, precision, ...).
    """
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    predictor = train_predictor(
        episodes, embeddings, DATASETS[dataset].step, use_actions=not blind, device=device or _device(), seed=seed,
        **options,
    )
    run = run_folder(dataset, encoder, root, seed)
    run.mkdir(parents=True, exist_ok=True)
    predictor.save(run / _predictor_file(blind))


def evaluate(dataset: str, encoder: str, root: str | Path | None = None, seed: int = 0) -> Report:
    """Score the static baseline and whichever predictors of the run trained with `seed` exist.
    The held-out windows and the shuffled actions are the same for every seed."""
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    spec = DATASETS[dataset]
    run = run_folder(dataset, encoder, root, seed)
    trained = {}
    for role, blind in (("predictor", False), ("blind", True)):
        path = run / _predictor_file(blind)
        if path.exists():
            trained[role] = Predictor.load(path, _device()).rollout
    report = evaluate_episodes(episodes, embeddings, spec.step, spec.move_threshold, **trained)
    report.save(run, title=f"{dataset}, {encoder} encoder" + (f", seed {seed}" if seed else ""))
    return report


def compare(dataset: str, root: str | Path | None = None) -> str:
    """Compare the encoders whose predictors have been evaluated: plot error vs horizon for
    each, pick the winning encoder and write it to winner.json. Returns the winner."""
    folder = data_root(root) / dataset
    reports = {}
    for results in sorted(folder.glob("*/results.json")):
        report = Report.load(results.parent)
        if PREDICTOR in report.errors:
            reports[results.parent.name] = report
    assert reports, f"no evaluated predictor under {folder}"
    winner = winning_encoder(reports)
    plot_errors(reports, folder / "error-vs-horizon.png")
    at_gate = {name: r.errors[PREDICTOR]["moving"][r.horizons.index(GATE_HORIZON)] for name, r in reports.items()}
    (folder / "winner.json").write_text(
        json.dumps(
            {
                "encoder": winner,
                "rule": f"lowest predictor error at horizon {GATE_HORIZON} on held-out moving windows",
                "predictor_error": at_gate,
                "validation_gate": {name: r.validation_gate for name, r in reports.items()},
            },
            indent=2,
        )
    )
    return winner


def seeds(dataset: str, encoder: str, seeds: list[int], root: str | Path | None = None) -> Path:
    """Compare the evaluated runs of one encoder trained with different seeds: write their
    gate verdicts and error spread to seeds.md, and plot one row per seed. Returns the path
    of seeds.md."""
    folder = data_root(root) / dataset / encoder
    reports = {seed: Report.load(run_folder(dataset, encoder, root, seed)) for seed in seeds}
    name = ENCODER_NAMES.get(encoder, encoder)
    plot_errors({f"{name}, seed {seed}": r for seed, r in reports.items()}, folder / "error-vs-horizon-seeds.png")
    path = folder / "seeds.md"
    path.write_text(seeds_to_markdown(reports, f"{dataset}, {encoder} encoder"), encoding="utf-8")
    return path


def retrievals(dataset: str, encoder: str, root: str | Path | None = None, n_rollouts: int = 3, seed: int = 0) -> Path:
    """Nearest-real-frame retrievals for a few held-out rollouts, drawn into retrievals.png.

    The rollouts start from held-out windows picked at random (by `seed`) among those where
    the object moves within the gate's horizon. Each prediction is shown as the training
    frame whose embedding is nearest to it, next to the true frame, with how far the object
    in the retrieved frame is from the true object position.
    """
    folder = data_root(root) / dataset
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    spec = DATASETS[dataset]
    train_set = [(ep, e) for ep, e in zip(episodes, embeddings) if ep.split == "train"]
    held_out = [(ep, e) for ep, e in zip(episodes, embeddings) if ep.split == "held-out"]
    held_episodes, held_embeddings = [ep for ep, _ in held_out], [e for _, e in held_out]

    windows = make_windows(held_episodes, spec.step, future=max(HORIZONS))
    c = windows.context
    moved = np.linalg.norm(windows.object_pos[:, c : c + GATE_HORIZON] - windows.object_pos[:, c - 1 : c], axis=-1)
    moving = np.flatnonzero(moved.max(axis=1) > spec.move_threshold)
    picked = np.sort(np.random.default_rng(seed).choice(moving, min(n_rollouts, len(moving)), replace=False))
    columns = [c + h - 1 for h in HORIZONS]  # window positions of the frames at each horizon

    models = {}
    for label, blind in (("predictor", False), ("action-blind", True)):
        path = folder / encoder / _predictor_file(blind)
        if path.exists():
            models[label] = Predictor.load(path, _device())

    rows = []
    for w in picked:
        ep = held_episodes[windows.episode[w]]
        frames = windows.frames[w]
        rows.append(
            {"label": f"{ep.name}\ntrue", "frames": [ep.frames[frames[c - 1]]] + [ep.frames[frames[k]] for k in columns],
             "notes": [""] * (1 + len(HORIZONS))}
        )
        for label, model in models.items():
            predicted = model.rollout(
                windows.gather(held_embeddings, [w], slice(None, c)), windows.pusher_pos[[w], :c], windows.actions[[w], c - 1 :]
            )[0]
            found = nearest_frames(predicted[[h - 1 for h in HORIZONS]], [e for _, e in train_set],
                                   std=model.emb_std.cpu().numpy())
            images, notes = [None], [""]
            for (e, f), k in zip(found, columns):
                images.append(train_set[e][0].frames[f])
                off = np.linalg.norm(train_set[e][0].object_pos[f] - windows.object_pos[w, k])
                notes.append(f"object off by {off:.0f}")
            rows.append({"label": f"{label}\n(nearest)", "frames": images, "notes": notes})

    path = folder / encoder / "retrievals.png"
    save_retrievals(rows, HORIZONS, path, held_episodes[0].unit)
    return path


def rank(
    dataset: str, encoder: str, root: str | Path | None = None, n_candidates: int = 32, horizon: int = 8, seed: int = 0
) -> RankingReport:
    """The ranking test on Push-T, judged by Push-T's own simulator. Writes ranking/ in the
    encoder's folder: ranking.json, report.md and ranking.png.

    One starting state per held-out episode: a frame drawn by `seed` among those with two
    steps of history (the predictor's context) and `horizon` steps of future in which the T
    really moves. From each, `n_candidates` candidate paths perturb the recorded pusher path
    over the next `horizon` steps (candidate 0 is the recorded path). The predictor rolls
    each candidate out from the real context frames; the probe reads its final T centre;
    the score is that centre's distance to the goal. The judge replays the recorded path up
    to the start (so the T starts exactly where it was, moving as it was), then the
    candidate. The control rolls each candidate out with the actions of another candidate
    path (a fixed permutation over all candidates of all starts).
    """
    assert dataset == "pusht", "the ranking test needs Push-T's judge and goal"
    from pwm.datasets import pusht
    from pwm.replay import Push, get_replay

    folder = data_root(root) / dataset
    gate = Report.load(folder / encoder).validation_gate
    assert gate, f"the ranking test runs only if the validation gate passed ({encoder}: {gate})"
    episodes = load_episodes(folder / "episodes")
    embeddings = load_embeddings(dataset, encoder, episodes, root)
    spec = DATASETS[dataset]
    train_ids = [i for i, ep in enumerate(episodes) if ep.split == "train"]
    probe = Probe.fit([embeddings[i] for i in train_ids], [episodes[i].object_pos for i in train_ids])
    predictor = Predictor.load(folder / encoder / _predictor_file(blind=False), _device())
    poses = pusht.block_poses(folder / "raw")
    judge = get_replay("pusht")
    goal = pusht.goal_centre()
    rng = np.random.default_rng(seed)
    stride = int(round(spec.step * pusht.FPS))  # frames per step
    span = horizon * stride

    starts, contexts, positions, candidates, plans = [], [], [], [], []
    for ep, emb in zip(episodes, embeddings):
        if ep.split != "held-out":
            continue
        frames = np.arange((CONTEXT - 1) * stride, len(ep.frames) - span)
        moved = np.array([np.linalg.norm(ep.object_pos[t + stride : t + span + 1 : stride] - ep.object_pos[t], axis=1).max()
                          for t in frames])
        frames = frames[moved > spec.move_threshold]
        if not len(frames):
            continue
        t = int(rng.choice(frames))
        context = t - stride * np.arange(CONTEXT - 1, -1, -1)  # frames of the context
        paths = candidate_paths(ep.pusher_pos[t : t + span + 1], n_candidates, seed=seed + len(starts),
                                workspace=ep.workspace)
        starts.append(f"{ep.name} frame {t}")
        contexts.append(np.asarray(emb[context]))
        positions.append(ep.pusher_pos[context])
        candidates.append(paths)
        plans.append((ep, t))

    actions = np.stack([np.diff(paths[:, ::stride], axis=1) for paths in candidates])  # (S, N, H, 2)
    shuffled = actions.reshape(-1, horizon, 2)[np.random.default_rng(seed).permutation(actions.shape[0] * n_candidates)]
    shuffled = shuffled.reshape(actions.shape)

    def final_centres(run_actions):
        out = []
        for context, pos, a in zip(contexts, positions, run_actions):
            n = len(a)
            predicted = predictor.rollout(np.repeat(context[None], n, 0), np.repeat(pos[None], n, 0), a)
            out.append(probe.predict(predicted[:, -1]))
        return np.array(out)  # (S, N, 2)

    predicted, control = final_centres(actions), final_centres(shuffled)

    judged, judge_check = [], []
    for (ep, t), paths in zip(plans, candidates):
        finals = []
        for path in paths:
            push = Push(path=np.concatenate([ep.pusher_pos[:t], path]),
                        timestamps=np.concatenate([ep.timestamps[:t], ep.timestamps[t] + np.arange(len(path)) / pusht.FPS]),
                        start=poses[ep.name][0], shape="T", unit=ep.unit, workspace=ep.workspace)
            finals.append(judge.replay(push).final[:2])
        judged.append(finals)
        judge_check.append(float(np.linalg.norm(finals[0] - ep.object_pos[t + span])))
    judged = np.array(judged)
    now = np.array([ep.object_pos[t] for ep, t in plans])[:, None]

    report = RankingReport(
        unit=episodes[0].unit, step=spec.step, horizon=horizon, seed=seed, starts=starts,
        predicted=np.linalg.norm(predicted - goal, axis=-1).tolist(),
        control=np.linalg.norm(control - goal, axis=-1).tolist(),
        judged=np.linalg.norm(judged - goal, axis=-1).tolist(),
        judge_check=judge_check,
        final_error=np.linalg.norm(predicted - judged, axis=-1).tolist(),
        static_error=np.linalg.norm(now - judged, axis=-1).tolist(),
    )
    out = folder / encoder / "ranking"
    report.save(out)
    plot_ranking(report, out / "ranking.png", ENCODER_NAMES.get(encoder, encoder))
    return report


def run_folder(dataset: str, encoder: str, root: str | Path | None = None, seed: int = 0) -> Path:
    """Where the predictors and results of the run trained with `seed` live."""
    folder = data_root(root) / dataset / encoder
    return folder / f"seed-{seed}" if seed else folder


def _predictor_file(blind: bool) -> str:
    return "predictor-blind.pt" if blind else "predictor.pt"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
