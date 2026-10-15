"""Ranking test: can the predictor tell which candidate paths push the object closer to a goal?

From a recorded starting state, candidate pusher paths are made by perturbing the recorded
one. The predictor rolls each candidate out from the real context frames, and the probe
reads where it expects the object to end up; that position's distance to the goal is the
candidate's score. A simulator (the judge) replays every candidate, and the report says
how well the two rankings agree.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Candidate paths: the recorded path's displacement from the current pusher position is
# rotated, scaled and bent (an offset that grows linearly to its full size at the end).
MAX_TURN = np.radians(45)
SCALE = (0.5, 1.5)
BEND_STD = 30.0  # table units, per axis, at the end of the path
MARGIN = 20.0  # table units kept free along each wall (Push-T: walls at 5, pusher radius 15)


def candidate_paths(path: np.ndarray, n: int, seed: int, workspace: float) -> np.ndarray:
    """`n` candidate paths (n, K, 2) from a recorded pusher path (K, 2). The first is the
    recorded path itself; every candidate starts where the recorded one does."""
    rng = np.random.default_rng(seed)
    turn = rng.uniform(-MAX_TURN, MAX_TURN, n - 1)
    scale = rng.uniform(*SCALE, n - 1)
    bend = rng.normal(0, BEND_STD, (n - 1, 2))

    start, offset = path[0], path - path[0]
    progress = np.linspace(0, 1, len(path))[None, :, None]
    cos, sin = np.cos(turn), np.sin(turn)
    rotation = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)  # (n - 1, 2, 2)
    moved = scale[:, None, None] * np.einsum("nij,kj->nki", rotation, offset) + progress * bend[:, None]
    perturbed = np.clip(start + moved, MARGIN, workspace - MARGIN)
    perturbed[:, 0] = start
    return np.concatenate([path[None], perturbed])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation: +1 when `a` and `b` put the candidates in the same order,
    -1 in the reverse order. Tied values share their average rank. NaN if either is constant."""
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    norm = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / norm) if norm > 0 else float("nan")


@dataclass
class RankingReport:
    """Scores of every candidate path (row: starting state, column: candidate; candidate 0 is
    the recorded path). Every score is a distance from the object's final position to the
    goal, in table units: lower is better."""

    unit: str
    step: float  # seconds
    horizon: int  # steps each candidate path lasts
    seed: int  # draws the random pick
    starts: list[str]
    predicted: list[list[float]]  # read by the probe after the predictor's rollout
    control: list[list[float]]  # the same, rolled out with the actions of another candidate path
    judged: list[list[float]]  # the judge's (the simulator's) final object position
    judge_check: list[float] = field(default_factory=list)  # per start: judge vs recorded final position, recorded path
    final_error: list[list[float]] = field(default_factory=list)  # predicted vs judged final object position
    static_error: list[list[float]] = field(default_factory=list)  # current vs judged final object position

    def summary(self) -> dict:
        judged = np.array(self.judged)
        n_starts, n_candidates = judged.shape
        rows = np.arange(n_starts)
        random = judged[rows, np.random.default_rng(self.seed).integers(0, n_candidates, n_starts)]

        def ranking(scores):
            scores = np.array(scores)
            rho = np.array([spearman(s, j) for s, j in zip(scores, judged)])
            defined = rho[~np.isnan(rho)]
            top = judged[rows, scores.argmin(axis=1)]
            return dict(
                spearman_per_start=rho.tolist(),
                spearman_mean=float(defined.mean()) if len(defined) else float("nan"),
                spearman_median=float(np.median(defined)) if len(defined) else float("nan"),
                spearman_se=float(defined.std(ddof=1) / np.sqrt(len(defined))) if len(defined) > 1 else float("nan"),
                positive_starts=int((defined > 0).sum()),
                top_pick=float(top.mean()),
                top_pick_per_start=top.tolist(),
                beats_random_pick=int((top < random).sum()),
                loses_to_random_pick=int((top > random).sum()),
            )

        return dict(
            starts=n_starts,
            candidates=n_candidates,
            undefined_starts=int(sum(np.ptp(j) == 0 for j in judged)),
            predictor=ranking(self.predicted),
            control=ranking(self.control),
            random_pick=float(random.mean()),
            random_pick_per_start=random.tolist(),
            mean_candidate=float(judged.mean()),
            recorded_path=float(judged[:, 0].mean()),
            best_possible=float(judged.min(axis=1).mean()),
        )

    def save(self, folder: Path) -> None:
        """Write ranking.json (the scores and the summary) and report.md into `folder`."""
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ranking.json").write_text(json.dumps({**asdict(self), "summary": self.summary()}, indent=1))
        (folder / "report.md").write_text(self.to_markdown())

    @classmethod
    def load(cls, folder: Path) -> "RankingReport":
        results = json.loads((folder / "ranking.json").read_text())
        results.pop("summary", None)
        return cls(**results)

    def to_markdown(self) -> str:
        s = self.summary()
        p, c = s["predictor"], s["control"]
        defined = s["starts"] - s["undefined_starts"]

        def rho(r):
            return f"{r['spearman_mean']:+.2f} ± {r['spearman_se']:.2f} (median {r['spearman_median']:+.2f})"

        def row(cells):
            return "| " + " | ".join(cells) + " |"

        lines = [
            "# Ranking test",
            "",
            f"{s['starts']} starting states, {s['candidates']} candidate paths each (candidate 0 is the recorded "
            f"path), {self.horizon} steps of {self.step:g} s. Every number is a distance from the object's final "
            f"position to the goal, in {self.unit}, or a rank correlation between such distances.",
            "",
            f"Spearman rank correlation between the predictor's and the judge's distances, per starting state, "
            f"then averaged (mean ± standard error) over the {defined} starts where the judge's distances differ "
            f"({s['undefined_starts']} starts where every candidate ends the same are left out).",
            "",
            row(["", "Spearman", "starts with Spearman > 0", "judge's final distance of the top pick",
                 "top pick vs random pick: better / worse"]),
            row(["---", "---:", "---:", "---:", "---:"]),
            row(["predictor", rho(p), f"{p['positive_starts']} / {defined}", f"{p['top_pick']:.1f}",
                 f"{p['beats_random_pick']} / {p['loses_to_random_pick']}"]),
            row(["predictor, shuffled actions (control)", rho(c), f"{c['positive_starts']} / {defined}",
                 f"{c['top_pick']:.1f}", f"{c['beats_random_pick']} / {c['loses_to_random_pick']}"]),
            "",
            "Judge's final distance to the goal, mean over starting states:",
            "",
            row(["pick", f"distance ({self.unit})"]),
            row(["---", "---:"]),
            row(["predictor's top pick", f"{p['top_pick']:.1f}"]),
            row(["top pick, shuffled actions (control)", f"{c['top_pick']:.1f}"]),
            row([f"random pick (seed {self.seed})", f"{s['random_pick']:.1f}"]),
            row(["mean over all candidates (expected random pick)", f"{s['mean_candidate']:.1f}"]),
            row(["recorded path", f"{s['recorded_path']:.1f}"]),
            row(["best candidate (the judge's own top pick)", f"{s['best_possible']:.1f}"]),
        ]
        if self.final_error:
            lines += [
                "",
                f"How far the predicted final object position is from the judge's, mean over all candidates: "
                f"{np.mean(self.final_error):.1f} {self.unit} (if the object stayed where it is: "
                f"{np.mean(self.static_error):.1f}).",
            ]
        if self.judge_check:
            lines += [
                "",
                f"Judge check: the recorded path, replayed from the episode's start, ends "
                f"{np.median(self.judge_check):.2f} {self.unit} (median; max {np.max(self.judge_check):.2f}) from the "
                f"recorded final object position.",
            ]
        lines += [
            "",
            "## Per starting state",
            "",
            row(["start", "Spearman", "Spearman, shuffled", "top pick", "random pick", "best", "worst"]),
            row(["---"] + ["---:"] * 6),
        ]
        judged = np.array(self.judged)
        for i, name in enumerate(self.starts):
            lines.append(row([
                name, f"{p['spearman_per_start'][i]:+.2f}", f"{c['spearman_per_start'][i]:+.2f}",
                f"{p['top_pick_per_start'][i]:.1f}", f"{s['random_pick_per_start'][i]:.1f}",
                f"{judged[i].min():.1f}", f"{judged[i].max():.1f}",
            ]))
        return "\n".join(lines) + "\n"


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties given the average of the ranks they span."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        ranks[tied] = ranks[tied].mean()
    return ranks
