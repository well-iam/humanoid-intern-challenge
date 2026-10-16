"""Figures of the results report: object position error against horizon, and retrievals."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window; works headless on Colab
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pwm.evaluator import (  # noqa: E402
    REAL,
    TRUE,
    BLIND,
    GATE_1_MARK,
    GATE_HORIZON,
    GATE_MARGIN,
    PREDICTOR,
    PROBE_FLOOR,
    REVERSED,
    SHUFFLED,
    STATIC,
    NearMiss,
    Report,
)

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Categorical slots in their fixed order; each row keeps its colour in every panel.
COLOURS = {PREDICTOR: "#2a78d6", BLIND: "#eb6834", STATIC: "#1baf7a", REVERSED: "#eda100", SHUFFLED: "#e87ba4"}
FLOOR_COLOUR = "#c3c2b7"
LABELS = {
    PREDICTOR: "predictor",
    BLIND: "action-blind predictor",
    STATIC: "static baseline",
    REVERSED: "predictor, reversed actions",
    SHUFFLED: "predictor, shuffled actions",
    PROBE_FLOOR: "probe floor (real future frame)",
}
ENCODER_NAMES = {"dinov2": "DINOv2-small", "vjepa2": "V-JEPA 2 ViT-L"}  # for titles; others as named
LEGEND_ORDER = [PREDICTOR, BLIND, STATIC, REVERSED, SHUFFLED, PROBE_FLOOR]
# (title, window subset, rows, whether to mark the validation gate's pass mark)
PANELS = [
    ("Moving windows", "moving", [STATIC, BLIND, PREDICTOR, PROBE_FLOOR], True),
    ("All windows", "all", [STATIC, BLIND, PREDICTOR, PROBE_FLOOR], False),
    ("Wrong-action test, moving windows", "moving", [STATIC, REVERSED, SHUFFLED, PREDICTOR], False),
]


def plot_errors(reports: dict[str, Report], path: Path) -> None:
    """One row per encoder: error vs horizon on moving windows, on all windows, and with
    wrong actions. All panels share the error axis so encoders compare at a glance."""
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig = plt.figure(figsize=(11, 3.4 * len(reports) + 0.6), layout="constrained", facecolor=SURFACE)
    first = None
    for subfig, (encoder, report) in zip(np.atleast_1d(fig.subfigures(len(reports), 1)), reports.items()):
        subfig.set_facecolor(SURFACE)
        verdict = {True: "passes", False: "fails", None: "not run"}[report.validation_gate]
        subfig.suptitle(f"{ENCODER_NAMES.get(encoder, encoder)}: validation gate {verdict}", x=0.01, ha="left",
                        fontsize=11, fontweight="bold", color=INK)
        axes = subfig.subplots(1, len(PANELS))
        for column, (ax, (title, subset, names, pass_mark)) in enumerate(zip(axes, PANELS)):
            if first is None:
                first = ax
            else:
                ax.sharey(first)
            _style(ax, report)
            ax.tick_params(labelleft=column == 0)
            ax.set_title(title, color=INK_2, fontsize=9, loc="left")
            for name in names:
                if name in report.errors:
                    _line(ax, report, name, subset)
            if pass_mark and BLIND in report.errors:
                _pass_mark(ax, report)
            ax.set_xlabel(f"horizon (steps of {report.step:g} s)", color=INK_2)
        axes[0].set_ylabel(f"object position error ({report.unit})", color=INK_2)
    first.set_ylim(bottom=0)

    handles = {}
    for ax in fig.get_axes():
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    order = [LABELS[name] for name in LEGEND_ORDER if LABELS[name] in handles]
    fig.legend([handles[k] for k in order], order, loc="outside upper center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _style(ax, report: Report | NearMiss) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(report.horizons, [str(h) for h in report.horizons])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(FLOOR_COLOUR)
    ax.tick_params(colors=MUTED, length=0, labelcolor=INK_2)


def _line(ax, report: Report, name: str, subset: str) -> None:
    errors = report.errors[name][subset]
    if name == PROBE_FLOOR:
        ax.plot(report.horizons, errors, color=FLOOR_COLOUR, linewidth=2, label=LABELS[name], zorder=1)
        return
    ax.plot(
        report.horizons,
        errors,
        color=COLOURS[name],
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=LABELS[name],
        zorder=3 if name == PREDICTOR else 2,
        solid_capstyle="round",
    )


def _pass_mark(ax, report: Report) -> None:
    """The validation gate's pass mark at its horizon: 20% below the better baseline."""
    i = report.horizons.index(GATE_HORIZON)
    mark = (1 - GATE_MARGIN) * min(report.errors[STATIC]["moving"][i], report.errors[BLIND]["moving"][i])
    ax.plot([GATE_HORIZON / 1.15, GATE_HORIZON * 1.15], [mark, mark], color=INK, linewidth=1.5, zorder=4)
    ax.annotate(
        "pass mark", xy=(GATE_HORIZON * 1.22, mark), va="center", fontsize=8, color=INK_2, annotation_clip=False
    )


def plot_near_miss(test: NearMiss, path: Path, title: str) -> None:
    """The near-miss test: mean object displacement by horizon on push windows (left) and on
    near-miss windows (right), for the tracked object, the real future frames and each model."""
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained", facecolor=SURFACE, sharey=True)
    fig.suptitle(f"Near-miss test, {title}: does the object move only when the pusher touches it?", x=0.01,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    labels = {TRUE: "tracked object", REAL: "probe on the real future frame", PREDICTOR: LABELS[PREDICTOR],
              BLIND: LABELS[BLIND]}
    titles = {"push": "Push windows: the object moves",
              "near-miss": "Near-miss windows: the pusher moves, the object does not"}
    for ax, kind in zip(axes, ("push", "near-miss")):
        _style(ax, test)
        ax.set_title(titles[kind], color=INK_2, fontsize=9, loc="left")
        for row, values in test.displacement[kind].items():
            if row == TRUE:
                ax.plot(test.horizons, values, color=INK, linewidth=1.5, linestyle="--", label=labels[row], zorder=4)
            elif row == REAL:
                ax.plot(test.horizons, values, color=FLOOR_COLOUR, linewidth=2, label=labels[row], zorder=1)
            else:
                ax.plot(test.horizons, values, color=COLOURS[row], linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=labels[row], zorder=3)
        counts = " / ".join(str(n) for n in test.n_windows[kind])
        ax.set_xlabel(f"horizon (steps of {test.step:g} s)\nheld-out windows per horizon: {counts}", color=INK_2)
    axes[0].set_ylabel(f"object displacement ({test.unit})", color=INK_2)
    axes[0].set_ylim(bottom=0)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ranking(report, path: Path, encoder: str) -> None:
    """The ranking test: the predictor's score of every candidate against the judge's
    distance (left), and the rank correlation per starting state with the control (right)."""
    s = report.summary()
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained", facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle(f"Ranking test on Push-T, {encoder}: {s['starts']} starting states x {s['candidates']} candidate paths",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)

    judged, predicted = np.ravel(report.judged), np.ravel(report.predicted)
    top = max(judged.max(), predicted.max()) * 1.05
    _plain(left)
    left.plot([0, top], [0, top], color=FLOOR_COLOUR, linewidth=1.5, zorder=1, label="predicted = judged")
    left.scatter(judged, predicted, s=10, color=COLOURS[PREDICTOR], alpha=0.35, linewidths=0, zorder=2,
                 label="one candidate path")
    left.set_xlim(0, top), left.set_ylim(0, top)
    left.set_aspect("equal")
    left.set_xlabel(f"judge: final distance of the T to the goal ({report.unit})", color=INK_2)
    left.set_ylabel(f"predictor: expected final distance ({report.unit})", color=INK_2)
    left.set_title("Every candidate path", color=INK_2, fontsize=9, loc="left")
    left.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    _plain(right)
    rows = [(PREDICTOR, "predictor", s["predictor"]), (SHUFFLED, "predictor,\nshuffled actions", s["control"])]
    jitter = np.random.default_rng(0)
    for y, (name, label, r) in enumerate(rows):
        rho = np.array(r["spearman_per_start"])
        rho = rho[~np.isnan(rho)]
        right.scatter(rho, y + jitter.uniform(-0.18, 0.18, len(rho)), s=22, color=COLOURS[name], alpha=0.7,
                      linewidths=0, zorder=2)
        right.errorbar(r["spearman_mean"], y, xerr=1.96 * r["spearman_se"], fmt="o", color=INK, markersize=7,
                       markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0, linewidth=2, zorder=3)
        right.annotate(f"mean {r['spearman_mean']:+.2f}", xy=(r["spearman_mean"], y + 0.24), ha="center",
                       va="top", fontsize=8, color=INK)
    right.axvline(0, color=FLOOR_COLOUR, linewidth=1.5, zorder=1)
    right.set_yticks(range(len(rows)), [label for _, label, _ in rows])
    right.set_ylim(len(rows) - 0.5, -0.6)
    right.set_xlim(-1.05, 1.05)
    right.set_xlabel("Spearman rank correlation with the judge, per starting state", color=INK_2)
    right.set_title("Rank agreement (black: mean, 95% interval)", color=INK_2, fontsize=9, loc="left")
    right.grid(axis="x", color=GRID, linewidth=0.8)
    right.grid(axis="y", visible=False)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_probe_scaling(curve: dict, path: Path) -> None:
    """Probe error (cross-validated on training clips) against the number of clips the probe is fitted on, with
    Gate 1's pass mark."""
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained", facecolor=SURFACE)
    encoder = ENCODER_NAMES.get(curve["encoder"], curve["encoder"])
    fig.suptitle(f"Probe error against training clips, {encoder} on {curve['dataset']}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    _plain(ax)
    clips = [p["clips"] for p in curve["points"]]
    errors = [p["error"] for p in curve["points"]]
    mark = GATE_1_MARK * curve["workspace"]
    ax.axhline(mark, color=INK_2, linewidth=1, linestyle="--", zorder=1)
    ax.annotate(f"Gate 1 pass mark, {mark:g} {curve['unit']}", xy=(clips[0], mark), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.plot(clips, errors, color=COLOURS[PREDICTOR], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    for n, e in zip(clips, errors):
        ax.annotate(f"{e:.2f}", xy=(n, e), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                    color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(clips, [str(n) for n in clips])
    ax.minorticks_off()
    ax.set_ylim(0, max(errors) * 1.15)
    ax.set_xlabel(f"training clips the probe is fitted on ({curve['folds']}-fold cross-validation over "
                  f"{curve['n_training_clips']} training clips)", color=INK_2)
    ax.set_ylabel(f"probe error on left-out clips ({curve['unit']})", color=INK_2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plain(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FLOOR_COLOUR)
    ax.tick_params(colors=MUTED, length=0, labelcolor=INK_2)


def save_retrievals(rows: list[dict], horizons: list[int], path: Path, unit: str) -> None:
    """A grid of frames: per held-out rollout, the true frames and the nearest real frame to
    each prediction. `rows` = [{"label": str, "frames": [img now, img at each horizon],
    "notes": [str per column]}]."""
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8})
    columns = 1 + len(horizons)
    fig, axes = plt.subplots(len(rows), columns, figsize=(1.6 * columns + 1.4, 1.75 * len(rows)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, (ax_row, row) in enumerate(zip(axes, rows)):
        for c, (ax, image) in enumerate(zip(ax_row, row["frames"])):
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if image is None:
                ax.set_facecolor(SURFACE)
                continue
            ax.imshow(np.asarray(image), interpolation="nearest")
            if row["notes"][c]:
                ax.set_xlabel(row["notes"][c], color=INK_2, fontsize=7, labelpad=2)
            if r == 0:
                ax.set_title("now" if c == 0 else f"+{horizons[c - 1]} step{'s' if horizons[c - 1] > 1 else ''}",
                             color=INK_2, fontsize=8)
        ax_row[0].set_ylabel(row["label"], color=INK, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
