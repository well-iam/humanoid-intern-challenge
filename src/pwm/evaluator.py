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

The wrong-action test rolls the predictor out with reversed actions (each one pointing the
opposite way) and with shuffled actions (those of another held-out window). The model
integrates future pusher positions from whatever actions it is given.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pwm.episodes import Episode
from pwm.probe import Probe
from pwm.windows import make_windows

HORIZONS = [1, 2, 4, 8]
BATCH = 32  # windows per rollout call; real predictions are 8 x 256 x 1024 floats per window

# Report rows.
STATIC = "static baseline"
PREDICTOR = "predictor"
BLIND = "action-blind predictor"
REVERSED = "predictor, reversed actions"
SHUFFLED = "predictor, shuffled actions"
PROBE_FLOOR = "probe floor"  # real future frames through the probe: what a perfect predictor would get

# Pass marks, fixed before seeing any results.
GATE_1_MARK = 0.04  # probe error below 4% of the workspace width
GATE_HORIZON = 4  # validation gate: at horizon 4, on moving windows,
GATE_MARGIN = 0.2  # the predictor's error is at least 20% below the better baseline


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
    errors: dict[str, dict[str, list[float]]]  # row -> {"moving": [...], "all": [...]}, per horizon

    @property
    def gate_1(self) -> bool:
        """Gate 1: the probe finds the object in real embeddings."""
        return self.probe_error < GATE_1_MARK * self.workspace

    @property
    def validation_gate(self) -> bool | None:
        """At horizon 4 on moving windows, the predictor's error is at least 20% below the
        better of the static baseline and the action-blind predictor, and both wrong-action
        rows raise it. None if the report lacks any of these rows."""
        rows = (STATIC, PREDICTOR, BLIND, REVERSED, SHUFFLED)
        if GATE_HORIZON not in self.horizons or any(row not in self.errors for row in rows):
            return None
        i = self.horizons.index(GATE_HORIZON)
        e = {row: self.errors[row]["moving"][i] for row in rows}
        beats_baselines = e[PREDICTOR] <= (1 - GATE_MARGIN) * min(e[STATIC], e[BLIND])
        return bool(beats_baselines and e[REVERSED] > e[PREDICTOR] and e[SHUFFLED] > e[PREDICTOR])

    def save(self, folder: Path, title: str) -> None:
        """Write results.json (with the gate flags) and report.md into `folder`."""
        folder.mkdir(parents=True, exist_ok=True)
        results = {**asdict(self), "gate_1": self.gate_1, "validation_gate": self.validation_gate}
        (folder / "results.json").write_text(json.dumps(results, indent=2))
        (folder / "report.md").write_text(self.to_markdown(title))

    @classmethod
    def load(cls, folder: Path) -> "Report":
        results = json.loads((folder / "results.json").read_text())
        results.pop("gate_1", None), results.pop("validation_gate", None)
        return cls(**results)

    def to_markdown(self, title: str) -> str:
        def row(cells):
            return "| " + " | ".join(cells) + " |"

        def verdict(flag):
            return {True: "**pass**", False: "**fail**", None: "not run (needs both predictors)"}[flag]

        lines = [
            f"# Results: {title}",
            "",
            f"Probe error on held-out real frames: {self.probe_error:.2f} {self.unit} "
            f"({100 * self.probe_error / self.workspace:.1f}% of the workspace width, {self.workspace:g}).",
            "",
            f"- Gate 1 (probe error under {100 * GATE_1_MARK:g}% of the workspace width): {verdict(self.gate_1)}",
            f"- Validation gate (at horizon {GATE_HORIZON} on moving windows, the predictor's error is at least "
            f"{100 * GATE_MARGIN:g}% below the better baseline, and both wrong-action rows raise it): "
            f"{verdict(self.validation_gate)}",
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


def winning_encoder(reports: dict[str, Report]) -> str:
    """The encoder whose predictor has the lowest held-out error at the validation gate's
    horizon on moving windows. `reports` maps encoder name -> its report."""
    return min(reports, key=lambda name: _at_gate(reports[name], PREDICTOR))


def _at_gate(report: Report, row: str) -> float:
    return report.errors[row]["moving"][report.horizons.index(GATE_HORIZON)]


def nearest_frames(
    queries: np.ndarray, bank: list[np.ndarray], std: np.ndarray | None = None, chunk: int = 128
) -> list[tuple[int, int]]:
    """For each query frame embedding (Q, patches, dims), the (episode, frame) of the closest
    real frame embedding in `bank` (one (T, patches, dims) array per episode), by Euclidean
    distance over the whole patch grid. Predictions are never decoded to pixels; the nearest
    real frame shows what a predicted embedding looks like.

    `std` (dims,) divides each channel first, e.g. the predictor's standardisation, so the
    distance is measured where the predictor was trained. The bank can be memory-mapped.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = torch.as_tensor(1.0 if std is None else 1.0 / np.asarray(std), dtype=torch.float32, device=device)
    q = (torch.as_tensor(np.asarray(queries), dtype=torch.float32, device=device) * scale).flatten(1)
    best = torch.full((len(q),), torch.inf, device=device)
    found = torch.zeros((len(q), 2), dtype=torch.long, device=device)
    for e, embeddings in enumerate(bank):
        for start in range(0, len(embeddings), chunk):
            b = torch.as_tensor(np.array(embeddings[start : start + chunk], dtype=np.float32), device=device)
            distance, frame = torch.cdist(q, (b * scale).flatten(1)).min(dim=1)
            closer = distance < best
            best = torch.where(closer, distance, best)
            found[closer] = torch.stack([torch.full_like(frame, e), frame + start], dim=1)[closer]
    return [(int(e), int(f)) for e, f in found.cpu()]


def evaluate(
    episodes: list[Episode],
    embeddings: list[np.ndarray],
    step: float,
    move_threshold: float,
    predictor=None,
    blind=None,
    horizons: list[int] = HORIZONS,
    seed: int = 0,
) -> Report:
    """`embeddings[i]` holds the frame embeddings of `episodes[i]`, shape (T, patches, dims).

    `predictor` and `blind` (the action-blind predictor) are rollout functions; the static
    baseline is always included, and the wrong-action test runs whenever `predictor` is given.
    `move_threshold` is in table units: a window is moving at horizon h if the object moves
    further than this from its current position within h steps.
    """
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

    actions = windows.actions[:, c - 1 :]
    runs = {STATIC: (static_baseline, actions)}
    if predictor is not None:
        runs[PREDICTOR] = (predictor, actions)
    if blind is not None:
        runs[BLIND] = (blind, actions)
    if predictor is not None:
        runs[REVERSED] = (predictor, -actions)
        runs[SHUFFLED] = (predictor, actions[np.random.default_rng(seed).permutation(len(actions))])

    per_window = {}
    for name, (rollout, run_actions) in runs.items():
        errors = []
        for b in range(0, len(windows), BATCH):
            rows = slice(b, b + BATCH)
            predicted = rollout(
                windows.gather(held_out_embeddings, rows, slice(None, c)), windows.pusher_pos[rows, :c], run_actions[rows]
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
