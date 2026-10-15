"""Panda replay of recorded pusher paths: the simulated Panda pushes the object along each path.

Replays every held-out episode (no frames) and compares the simulated final object position
with the recorded one, then renders one episode as a GIF: the recording, the Panda seen from
above (arm drawn see-through) and from the side. Writes to <root>/<dataset>/panda/:
replay.json (per episode and summary) and <episode>.gif.

Push-T only for now: the start angle of the T comes from the raw file. My clips carry no
object angle yet.
"""

import argparse
import json

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pwm import stages
from pwm.datasets import pusht
from pwm.episodes import load_episodes
from pwm.replay import Push, get_replay

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="pusht")
parser.add_argument("--episode", help="episode for the GIF (default: the first held-out episode)")
parser.add_argument("--slowdown", type=float, default=1.0, help="replay this many times slower than recorded")
parser.add_argument("--floating", action="store_true", help="floating gripper instead of the arm")
parser.add_argument("--size", type=int, default=256, help="side of each GIF panel, pixels")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()
assert args.dataset == "pusht", "only Push-T episodes carry the object's start angle so far"

folder = stages.data_root(args.root) / args.dataset
episodes = [ep for ep in load_episodes(folder / "episodes") if ep.split == "held-out"]
poses = pusht.block_poses(folder / "raw")
options = dict(floating=args.floating, slowdown=args.slowdown, root=args.root)
panda = get_replay("panda", **options)


def push_of(ep):
    return Push(path=ep.pusher_pos, timestamps=ep.timestamps, start=poses[ep.name][0], shape="T",
                unit=ep.unit, workspace=ep.workspace)


rows = {}
for ep in episodes:
    result = panda.replay(push_of(ep))
    rows[ep.name] = dict(
        final_error=float(np.linalg.norm(result.final[:2] - ep.object_pos[-1])),
        moved=float(np.linalg.norm(ep.object_pos[-1] - ep.object_pos[0])),
        simulated_moved=float(np.linalg.norm(result.final[:2] - ep.object_pos[0])),
        tracking_error_max=float(np.linalg.norm(result.pusher - ep.pusher_pos, axis=1).max()),
        simulated_final=result.final[:2].tolist(),
        recorded_final=ep.object_pos[-1].tolist(),
    )
    print(f"{ep.name}: final error {rows[ep.name]['final_error']:6.1f}, T moved {rows[ep.name]['moved']:6.1f} "
          f"(simulated {rows[ep.name]['simulated_moved']:6.1f}), worst tracking {rows[ep.name]['tracking_error_max']:.2f}")


def summary(key):
    v = np.array([r[key] for r in rows.values()])
    return dict(median=float(np.median(v)), mean=float(v.mean()), p90=float(np.percentile(v, 90)), max=float(v.max()))


out = folder / "panda"
out.mkdir(parents=True, exist_ok=True)
report = dict(unit=pusht.UNIT, cm_per_unit=50 / pusht.WORKSPACE, **options, episodes=len(rows),
              **{key: summary(key) for key in ("final_error", "moved", "simulated_moved", "tracking_error_max")},
              per_episode=rows)
(out / "replay.json").write_text(json.dumps(report, indent=1))
print(f"\nover {len(rows)} held-out episodes ({pusht.UNIT}; 1 unit = {50 / pusht.WORKSPACE:.3f} cm in the simulator):")
for key in ("final_error", "moved", "simulated_moved", "tracking_error_max"):
    print(f"  {key:20s} " + "  ".join(f"{k} {v:.2f}" for k, v in report[key].items()))

ep = next(e for e in episodes if e.name == args.episode or args.episode is None)
top = panda.replay(push_of(ep), frame_size=args.size).frames
side = get_replay("panda", camera="side", **options).replay(push_of(ep), frame_size=args.size).frames
recorded = np.array([np.asarray(Image.fromarray(f).resize((args.size, args.size), Image.BILINEAR)) for f in ep.frames])
gap = np.full((len(top), args.size, 4, 3), 255, np.uint8)
frames = np.concatenate([recorded, gap, top, gap, side], axis=2)
header = Image.new("RGB", (frames.shape[2], 24), "white")
draw = ImageDraw.Draw(header)
for i, label in enumerate(["recorded (Push-T)", "Panda replay, from above", "Panda replay"]):
    draw.text((i * (args.size + 4) + args.size / 2, 12), label, fill="black", anchor="mm",
              font=ImageFont.load_default(size=14))
frames = np.concatenate([np.broadcast_to(np.asarray(header), (len(frames), *np.asarray(header).shape)), frames], axis=1)
step = ep.timestamps[1] - ep.timestamps[0]
iio.imwrite(out / f"{ep.name}.gif", frames, duration=1000 * step, loop=0)
row = rows[ep.name]
print(f"\n{ep.name}: simulated final {np.round(row['simulated_final'], 1)} vs recorded {np.round(row['recorded_final'], 1)} "
      f"-> {row['final_error']:.1f} units ({row['final_error'] * 50 / pusht.WORKSPACE:.1f} cm); GIF: {out / f'{ep.name}.gif'}")
