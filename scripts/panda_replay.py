"""Panda replay of recorded pusher paths: the simulated Panda pushes the object along each path.

Compares the simulated final object position with the recorded one, then renders GIFs: the recording, the
Panda seen from above (arm drawn see-through) and from the side. Writes to <root>/<dataset>/panda/:
replay.json (per episode and summary) and <episode>.gif.

- pusht: every held-out episode; the start angle of the T comes from the raw file.
- clips (my clips): every clip, its palm path moved by its contact shift (pwm.replay.palm), replayed from the
  first frame where the hand is found. Summaries per group: pushes with their own shift, pushes whose hand was
  found late (median shift), near-misses (median shift). The recording panel marks the palm (green) and the
  shifted point the gripper follows (blue).
"""

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pwm import stages
from pwm.datasets import pusht
from pwm.episodes import load_episodes
from pwm.replay import Push, get_replay, palm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="pusht", choices=["pusht", "clips"])
parser.add_argument("--episode", nargs="*", help="episodes for GIFs (default: the first held-out episode; "
                    "clips: the first held-out push with its own shift, and the first held-out near-miss)")
parser.add_argument("--slowdown", type=float, default=1.0, help="replay this many times slower than recorded")
parser.add_argument("--floating", action="store_true", help="floating gripper instead of the arm")
parser.add_argument("--size", type=int, default=256, help="side of each GIF panel, pixels")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

folder = stages.data_root(args.root) / args.dataset
options = dict(floating=args.floating, slowdown=args.slowdown, root=args.root)
panda = get_replay("panda", **options)
clips = args.dataset == "clips"

if clips:
    episodes = load_episodes(folder / "episodes")
    shifts = palm.shifts(episodes)

    def push_of(ep):
        return palm.clip_push(ep, shifts[ep.name].vector)

    def group(ep):
        if ep.kind == "near-miss":
            return "near-miss"
        return "push, own shift" if shifts[ep.name].own else "push, hand found late"
else:
    episodes = [ep for ep in load_episodes(folder / "episodes") if ep.split == "held-out"]
    poses = pusht.block_poses(folder / "raw")

    def push_of(ep):
        return Push(path=ep.pusher_pos, timestamps=ep.timestamps, start=poses[ep.name][0], shape="T",
                    unit=ep.unit, workspace=ep.workspace)

    def group(ep):
        return "held-out"

rows = {}
for ep in episodes:
    push = push_of(ep)
    result = panda.replay(push)
    row = dict(
        group=group(ep),
        split=ep.split,
        final_error=float(np.linalg.norm(result.final[:2] - ep.object_pos[-1])),
        moved=float(np.linalg.norm(ep.object_pos[-1] - push.start[:2])),
        simulated_moved=float(np.linalg.norm(result.final[:2] - push.start[:2])),
        tracking_error_max=float(np.linalg.norm(result.pusher - push.path, axis=1).max()),
        simulated_final=result.final[:2].tolist(),
        recorded_final=ep.object_pos[-1].tolist(),
    )
    if clips:
        hand = ep.hand_direction[ep.pusher_seen]
        turn = np.degrees(np.ptp(np.unwrap(np.arctan2(hand[:, 1], hand[:, 0]))))
        row.update(shift=shifts[ep.name].vector.tolist(), shift_length=float(np.linalg.norm(shifts[ep.name].vector)),
                   own_shift=shifts[ep.name].own, replay_from=float(push.timestamps[0]), hand_turn_deg=float(turn),
                   final_angle_error_deg=float(np.degrees(result.final[2] - ep.object_angle[-1])))
    rows[ep.name] = row
    print(f"{ep.name} [{row['group']}]: final error {row['final_error']:6.1f}, moved {row['moved']:6.1f} "
          f"(simulated {row['simulated_moved']:6.1f}), worst tracking {row['tracking_error_max']:.2f}")

KEYS = ("final_error", "moved", "simulated_moved", "tracking_error_max")


def summary(names, key):
    v = np.array([rows[n][key] for n in names])
    return dict(median=float(np.median(v)), mean=float(v.mean()), p90=float(np.percentile(v, 90)), max=float(v.max()))


groups = {g: [n for n in rows if rows[n]["group"] == g] for g in dict.fromkeys(r["group"] for r in rows.values())}
out = folder / "panda"
out.mkdir(parents=True, exist_ok=True)
unit = episodes[0].unit
summaries = {g: dict(episodes=len(names), **{key: summary(names, key) for key in KEYS}) for g, names in groups.items()}
report = dict(unit=unit, **options, episodes=len(rows))
if clips:
    report.update(gripper_half_cm=palm.GRIPPER_HALF, contact_frames_back=palm.BACK, groups=summaries)
else:  # the layout of ticket 06
    report.update(cm_per_unit=50 / pusht.WORKSPACE, **{key: summaries["held-out"][key] for key in KEYS})
report["per_episode"] = rows
(out / "replay.json").write_text(json.dumps(report, indent=1))
for g, names in groups.items():
    print(f"\n{g}: {len(names)} episodes ({unit})")
    for key in KEYS:
        print(f"  {key:20s} " + "  ".join(f"{k} {v:.2f}" for k, v in summaries[g][key].items()))


def default_gifs():
    held_out = [ep for ep in episodes if ep.split == "held-out"]
    if not clips:
        return [held_out[0].name]
    return [next(ep.name for ep in held_out if group(ep) == g) for g in ("push, own shift", "near-miss")]


def to_px(points, ep):
    return np.asarray(points) * args.size / ep.workspace


PALM, SHIFTED = (0, 210, 0), (20, 90, 255)  # the dots on my clip's frames


def save_gif(path, frames, step, keep=(PALM, SHIFTED)):
    """Each frame gets its own 254 colours plus the dots' colours: a GIF's usual per-frame palette merges the
    dots, only a few pixels each, into the colours around them."""
    images = []
    for frame in frames:
        image = Image.fromarray(np.ascontiguousarray(frame))
        palette = image.quantize(256 - len(keep)).getpalette()[: 3 * (256 - len(keep))]
        holder = Image.new("P", (1, 1))
        holder.putpalette(palette + [c for colour in keep for c in colour])
        images.append(image.quantize(palette=holder, dither=Image.Dither.NONE))
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 * step, loop=0)


for name in args.episode or default_gifs():
    ep = next(e for e in episodes if e.name == name)
    push = push_of(ep)
    first = len(ep.timestamps) - len(push.timestamps)
    top = panda.replay(push, frame_size=args.size).frames
    side = get_replay("panda", camera="side", **options).replay(push, frame_size=args.size).frames
    recorded = []
    for i, frame in enumerate(ep.frames[first:]):
        image = Image.fromarray(frame).resize((args.size, args.size), Image.BILINEAR)
        if clips:
            draw = ImageDraw.Draw(image)
            for point, colour in ((ep.pusher_pos[first + i], PALM), (push.path[i], SHIFTED)):
                x, y = to_px(point, ep)
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=colour, outline="white", width=2)
        recorded.append(np.asarray(image))
    recorded = np.array(recorded)
    gap = np.full((len(top), args.size, 4, 3), 255, np.uint8)
    frames = np.concatenate([recorded, gap, top, gap, side], axis=2)
    header = Image.new("RGB", (frames.shape[2], 24), "white")
    draw = ImageDraw.Draw(header)
    labels = ["my clip: palm green, shifted blue" if clips else "recorded (Push-T)", "Panda replay, from above",
              "Panda replay"]
    for i, label in enumerate(labels):
        draw.text((i * (args.size + 4) + args.size / 2, 12), label, fill="black", anchor="mm",
                  font=ImageFont.load_default(size=14))
    frames = np.concatenate([np.broadcast_to(np.asarray(header), (len(frames), *np.asarray(header).shape)), frames],
                            axis=1)
    save_gif(out / f"{ep.name}.gif", frames, ep.timestamps[1] - ep.timestamps[0])
    row = rows[ep.name]
    print(f"\n{ep.name} [{row['group']}]: simulated final {np.round(row['simulated_final'], 1)} vs recorded "
          f"{np.round(row['recorded_final'], 1)} -> {row['final_error']:.1f} {unit}; GIF: {out / f'{ep.name}.gif'}")
