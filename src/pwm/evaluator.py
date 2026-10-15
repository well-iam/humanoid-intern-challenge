"""Evaluator: scores predictions of future frame embeddings by the object position the probe reads from them.

The probe is fitted on real embeddings of training episodes only. Every model is then
rolled out on held-out windows, and its object position error (probe on the predicted
embedding vs the true object position) is averaged per horizon, over moving windows and
over all windows.

A model is a function `rollout(context, pusher_pos, actions) -> predictions`:
- context: (B, C, patches, dims) real frame embeddings of the context frames;
- pusher_pos: (B, C, 2) pusher positions at the context frames;
- actions: (B, H, 2) pusher actions for the H future steps;
- predictions: (B, H, patches, dims), the predicted frame embedding after each step.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pwm.episodes import Episode
from pwm.probe import Probe
from pwm.windows import make_windows

HORIZONS = [1, 2, 4, 8]
BATCH = 256

# Real future frame embeddings through the probe: the error a perfect predictor would get.
PROBE_FLOOR = "probe floor"
STATIC = "static baseline"


def static_baseline(context, pusher_pos, actions):
    """The prediction that nothing changes: the last real frame embedding, repeated."""
    return np.repeat(context[:, -1:], len(actions[0]), axis=1)


@dataclass
class Report:
    unit: str
    workspace: float
    step: float  # seconds
    move_threshold: float  # table units
    probe_error: float  # mean probe error on all held-out real frames
    horizons: list[int]
    n_windows: int  # held-out windows
    n_moving: list[int]  # moving windows, per horizon
    errors: dict[str, dict[str, list[float]]]  # model -> {"moving": [...], "all": [...]}, per horizon

    def save(self, folder: Path, title: str) -> None:
        """Write results.json and report.md into `folder`."""
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "results.json").write_text(json.dumps(asdict(self), indent=2))
        (folder / "report.md").write_text(self.to_markdown(title))

    @classmethod
    def load(cls, folder: Path) -> "Report":
        return cls(**json.loads((folder / "results.json").read_text()))

    def to_markdown(self, title: str) -> str:
        def row(cells):
            return "| " + " | ".join(cells) + " |"

        lines = [
            f"# Results: {title}",
            "",
            f"Probe error on held-out real frames: {self.probe_error:.2f} {self.unit} "
            f"({100 * self.probe_error / self.workspace:.1f}% of the workspace width, {self.workspace:g}).",
            "",
            f"Object position error in {self.unit}, mean over held-out windows, by horizon "
            f"(one step = {self.step:g} s). Moving windows: the object moves more than "
            f"{self.move_threshold:g} {self.unit} within the horizon. "
            f"The probe floor is the probe's own error on the real future frame.",
            "",
            row([""] + [f"{h} step{'s' if h > 1 else ''}" for h in self.horizons]),
            row(["---"] + ["---:"] * len(self.horizons)),
            row(["windows: moving / all"] + [f"{m} / {self.n_windows}" for m in self.n_moving]),
        ]
        for name, err in self.errors.items():
            for subset in ("moving", "all"):
                lines.append(row([f"{name}, {subset}"] + [f"{e:.2f}" for e in err[subset]]))
        return "\n".join(lines) + "\n"


def evaluate(
    episodes: list[Episode],
    embeddings: list[np.ndarray],
    step: float,
    move_threshold: float,
    models: dict | None = None,
    horizons: list[int] = HORIZONS,
) -> Report:
    """`embeddings[i]` holds the frame embeddings of `episodes[i]`, shape (T, patches, dims).

    `models` maps a name to a rollout function; the static baseline is always included.
    `move_threshold` is in table units: a window is moving at horizon h if the object moves
    further than this from its current position within h steps.
    """
    models = {STATIC: static_baseline, **(models or {})}
    train = [i for i, ep in enumerate(episodes) if ep.split == "train"]
    held_out = [ep for ep in episodes if ep.split == "held-out"]
    held_out_embeddings = [embeddings[i] for i, ep in enumerate(episodes) if ep.split == "held-out"]

    probe = Probe.fit([embeddings[i] for i in train], [episodes[i].object_pos for i in train])
    probe_pos = [probe.predict(e) for e in held_out_embeddings]
    probe_error = np.mean(np.concatenate([_dist(p, ep.object_pos) for p, ep in zip(probe_pos, held_out)]))

    windows = make_windows(held_out, step, future=max(horizons))
    c = windows.context
    now, future = windows.object_pos[:, c - 1 : c], windows.object_pos[:, c:]  # (N, 1, 2), (N, H, 2)
    moved = _dist(future, now)  # (N, H)
    moving = [moved[:, :h].max(axis=1) > move_threshold for h in horizons]

    per_window = {}
    actions = windows.actions[:, c - 1 :]
    for name, rollout in models.items():
        errors = []
        for b in range(0, len(windows), BATCH):
            rows = slice(b, b + BATCH)
            predicted = rollout(
                windows.gather(held_out_embeddings, rows, slice(None, c)), windows.pusher_pos[rows, :c], actions[rows]
            )
            n, steps = predicted.shape[:2]
            predicted_pos = probe.predict(predicted.reshape(n * steps, *predicted.shape[2:])).reshape(n, steps, 2)
            errors.append(_dist(predicted_pos, future[rows]))
        per_window[name] = np.concatenate(errors) if errors else np.zeros((0, max(horizons)))
    per_window[PROBE_FLOOR] = _dist(windows.gather(probe_pos, cols=slice(c, None)), future)

    return Report(
        unit=held_out[0].unit,
        workspace=held_out[0].workspace,
        step=step,
        move_threshold=move_threshold,
        probe_error=float(probe_error),
        horizons=list(horizons),
        n_windows=len(windows),
        n_moving=[int(m.sum()) for m in moving],
        errors={
            name: {
                "moving": [_mean(err[m, h - 1]) for h, m in zip(horizons, moving)],
                "all": [_mean(err[:, h - 1]) for h in horizons],
            }
            for name, err in per_window.items()
        },
    )


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")
