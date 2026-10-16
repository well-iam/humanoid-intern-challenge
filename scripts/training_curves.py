"""Training batch loss (logged every 100 steps by scripts/train.py) of the predictors on my clips, next to
Push-T's DINOv2 seeds. Reads the training logs in <root>/<dataset>/logs/ and writes
<root>/clips/dinov2/training-curves.png."""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pwm import stages  # noqa: E402
from pwm.plots import GRID, INK_2, SURFACE  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()
root = stages.data_root(args.root)

RUNS = {  # run -> role -> the log written by scripts/train.py
    "my clips": {"predictor": "clips/logs/train-predictor.log", "action-blind": "clips/logs/train-blind.log"},
    "Push-T seed 0": {"predictor": "pusht/logs/train-dinov2.log", "action-blind": "pusht/logs/train-dinov2-blind.log"},
    "Push-T seed 1": {"predictor": "pusht/logs/seeds/train-dinov2-seed1.log",
                      "action-blind": "pusht/logs/seeds/train-dinov2-blind-seed1.log"},
    "Push-T seed 2 (failed)": {"predictor": "pusht/logs/seeds/train-dinov2-seed2.log",
                               "action-blind": "pusht/logs/seeds/train-dinov2-blind-seed2.log"},
}
COLOURS = {"my clips": "#2a78d6", "Push-T seed 0": "#898781", "Push-T seed 1": "#c3c2b7",
           "Push-T seed 2 (failed)": "#eb6834"}


def curve(path):
    pairs = re.findall(r"step\s+(\d+)\s+loss\s+([\d.]+)", path.read_text())
    return [int(s) for s, _ in pairs], [float(v) for _, v in pairs]


plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True, facecolor=SURFACE)
fig.suptitle("Training batch loss, DINOv2-small, same recipe (2500 steps, lr 1e-3, batch 16, fp16)", x=0.01,
             ha="left", fontsize=11, fontweight="bold")
for ax, role in zip(axes, ("predictor", "action-blind")):
    ax.set_facecolor(SURFACE)
    for name, files in RUNS.items():
        if (root / files[role]).exists():
            steps, loss = curve(root / files[role])
            ax.plot(steps, loss, color=COLOURS[name], linewidth=2.5 if name == "my clips" else 1.5, label=name,
                    zorder=3 if name == "my clips" else 2)
    ax.set_title(role, loc="left", color=INK_2)
    ax.set_xlabel("training step")
    ax.grid(axis="y", color=GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("batch loss (smooth L1, standardised units)")
axes[0].legend(frameon=False)
out = root / "clips/dinov2/training-curves.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
