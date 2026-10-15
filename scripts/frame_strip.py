"""Visual check of an episode: a strip of frames with the pusher and object positions drawn on.

The pusher is a blue ring and the object a red cross, placed from the stored positions
(table units scaled to the image), so they should sit on the pusher and object in the frame.
Writes <root>/<dataset>/frame-strip.png.
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw

from pwm import stages
from pwm.episodes import load_episodes

SCALE = 3  # nearest-neighbour upscaling, so the markers stay readable on small frames
GAP = 4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--episode", help="episode name (default: the first held-out episode)")
parser.add_argument("--frames", type=int, default=8, help="number of frames, evenly spread over the episode")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

folder = stages.data_root(args.root) / args.dataset
episodes = load_episodes(folder / "episodes")
ep = next(e for e in episodes if e.name == args.episode or (args.episode is None and e.split == "held-out"))

picks = np.linspace(0, len(ep.frames) - 1, args.frames).round().astype(int)
h, w = ep.frames.shape[1:3]
strip = Image.new("RGB", (len(picks) * (w * SCALE + GAP) - GAP, h * SCALE + 20), "white")
draw = ImageDraw.Draw(strip)
for i, t in enumerate(picks):
    x0 = i * (w * SCALE + GAP)
    strip.paste(Image.fromarray(ep.frames[t]).resize((w * SCALE, h * SCALE), Image.NEAREST), (x0, 0))
    px, py = ep.pusher_pos[t] * w * SCALE / ep.workspace
    ox, oy = ep.object_pos[t] * w * SCALE / ep.workspace
    draw.ellipse((x0 + px - 6, py - 6, x0 + px + 6, py + 6), outline=(0, 90, 255), width=2)
    draw.line((x0 + ox - 7, oy, x0 + ox + 7, oy), fill=(230, 0, 0), width=2)
    draw.line((x0 + ox, oy - 7, x0 + ox, oy + 7), fill=(230, 0, 0), width=2)
    draw.text((x0 + 2, h * SCALE + 4), f"{ep.timestamps[t]:.1f} s", fill="black")

out = folder / "frame-strip.png"
strip.save(out)
print(f"{ep.name}: frames {picks.tolist()} -> {out}")
