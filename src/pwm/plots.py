"""Figures of the results report: object position error against horizon, and retrievals."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window; works headless on Colab
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pwm.evaluator import (  # noqa: E402
    BLIND,
    GATE_HORIZON,
    GATE_MARGIN,
    PREDICTOR,
    PROBE_FLOOR,
    REVERSED,
    SHUFFLED,
    STATIC,
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
LEGEND_ORDER = [PREDICTOR, BLIND, STATIC, REVERSED, SHUFFLED, PROBE_FLOOR]
PANELS = [
    ("Moving windows", "moving", [STATIC, BLIND, PREDICTOR, PROBE_FLOOR]),
    ("All windows", "all", [STATIC, BLIND, PREDICTOR, PROBE_FLOOR]),
    ("Wrong-action test, moving windows", "moving", [STATIC, REVERSED, SHUFFLED, PREDICTOR]),
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
        subfig.suptitle(f"{encoder}: validation gate {verdict}", x=0.01, ha="left", fontsize=11,
                        fontweight="semibold", color=INK)
        axes = subfig.subplots(1, len(PANELS))
        for ax, (title, subset, names) in zip(axes, PANELS):
            if first is None:
                first = ax
            else:
                ax.sharey(first)
            _style(ax, report)
            ax.set_title(title, color=INK_2, fontsize=9, loc="left")
            for name in names:
                if name in report.errors:
                    _line(ax, report, name, subset)
            if subset == "moving" and BLIND in report.errors and title.startswith("Moving"):
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


def _style(ax, report: Report) -> None:
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
        "pass mark", xy=(GATE_HORIZON * 1.17, mark), va="center", fontsize=8, color=INK_2, annotation_clip=False
    )


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
