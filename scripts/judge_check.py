"""Replay check of the Push-T judge: the judge's noise floor.

Each held-out Push-T episode's pusher path is replayed in Push-T's own simulator from the
recorded start pose of the T. The distance between the replayed and the recorded final T
centre is the judge's noise floor. Writes <root>/pusht/judge/noise-floor.json.
"""

import argparse
import json

import numpy as np

from pwm import stages
from pwm.datasets import pusht
from pwm.episodes import load_episodes
from pwm.replay import Push, get_replay

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

folder = stages.data_root(args.root) / "pusht"
episodes = [ep for ep in load_episodes(folder / "episodes") if ep.split == "held-out"]
poses = pusht.block_poses(folder / "raw")
judge = get_replay("pusht")

rows = {}
for ep in episodes:
    recorded = poses[ep.name]
    push = Push(path=ep.pusher_pos, timestamps=ep.timestamps, start=recorded[0],
                shape="T", unit=ep.unit, workspace=ep.workspace)
    final = judge.replay(push).final
    angle = (final[2] - recorded[-1, 2] + np.pi) % (2 * np.pi) - np.pi
    rows[ep.name] = dict(
        position_error=float(np.linalg.norm(final[:2] - recorded[-1, :2])),
        angle_error_deg=float(abs(np.degrees(angle))),
        moved=float(np.linalg.norm(recorded[-1, :2] - recorded[0, :2])),
    )


def summary(key):
    v = np.array([r[key] for r in rows.values()])
    return dict(median=np.median(v), mean=v.mean(), p90=np.percentile(v, 90), max=v.max())


result = dict(
    unit=pusht.UNIT,
    episodes=len(rows),
    final_position_error=summary("position_error"),
    final_angle_error_deg=summary("angle_error_deg"),
    distance_the_t_moved=summary("moved"),
    per_episode=rows,
)
out = folder / "judge"
out.mkdir(parents=True, exist_ok=True)
(out / "noise-floor.json").write_text(json.dumps(result, indent=1, default=float))

print(f"Push-T judge noise floor over {len(rows)} held-out episodes ({pusht.UNIT}):")
for key in ("final_position_error", "final_angle_error_deg", "distance_the_t_moved"):
    print(f"  {key:22s} " + "  ".join(f"{k} {v:.2f}" for k, v in result[key].items()))
