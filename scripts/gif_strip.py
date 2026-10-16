"""The GIF strip for the README: my clip | the world model's prediction | the Panda replaying my palm path.

The prediction panel shows, at each moment, what the predictor expected one second earlier: it rolls out
HORIZON steps (of 0.25 s) from the real frames before that, with the palm's real moves as the actions. A
prediction is an embedding, never decoded to pixels, so the panel shows the training frame (from other clips)
whose embedding is nearest to it. The Panda panel is the replay from above, as in scripts/panda_replay.py. Writes
<root>/clips/gif-strip/<clip>.gif.
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pwm import stages
from pwm.datasets import DATASETS
from pwm.episodes import load_episodes
from pwm.evaluator import nearest_frames
from pwm.predictor import Predictor
from pwm.replay import get_replay, palm
from pwm.windows import CONTEXT

HORIZON = 4  # steps ahead: 1 s
PALM = (0, 210, 0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode", nargs="*", help="clips (default: the first held-out push with its own shift)")
parser.add_argument("--encoder", default="dinov2")
parser.add_argument("--size", type=int, default=256, help="side of each panel, pixels")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

folder = stages.data_root(args.root) / "clips"
episodes = load_episodes(folder / "episodes")
embeddings = stages.load_embeddings("clips", args.encoder, episodes, args.root)
train = [(ep, e) for ep, e in zip(episodes, embeddings) if ep.split == "train"]
predictor = Predictor.load(folder / args.encoder / "predictor.pt", stages._device())
shifts = palm.shifts(episodes)
stride = int(round(DATASETS["clips"].step / (episodes[0].timestamps[1] - episodes[0].timestamps[0])))
names = args.episode or [next(ep.name for ep in episodes if ep.split == "held-out" and ep.kind == "push"
                              and shifts[ep.name].own)]
out = folder / "gif-strip"
out.mkdir(exist_ok=True)
font = ImageFont.load_default(size=14)

for name in names:
    i = next(i for i, ep in enumerate(episodes) if ep.name == name)
    ep, emb = episodes[i], embeddings[i]

    # The prediction for frame k, made from the real frames up to k - HORIZON steps.
    lead = HORIZON * stride  # frames between the last real frame used and the predicted one
    first_k = (CONTEXT - 1) * stride + lead
    targets = np.arange(first_k, len(ep.frames))
    context = targets[:, None] - lead - stride * np.arange(CONTEXT - 1, -1, -1)[None]  # (K, CONTEXT) frames
    path = targets[:, None] - lead + stride * np.arange(HORIZON + 1)[None]  # the palm from the last real frame on
    predicted = predictor.rollout(np.asarray(emb)[context], ep.pusher_pos[context], np.diff(ep.pusher_pos[path], axis=1))
    found = nearest_frames(predicted[:, -1], [e for _, e in train], std=predictor.emb_std.cpu().numpy())
    guess = {int(k): train[e][0].frames[f] for k, (e, f) in zip(targets, found)}

    push = palm.clip_push(ep, shifts[name].vector)
    start = len(ep.timestamps) - len(push.timestamps)  # the replay starts where the hand is first found
    panda = get_replay("panda", root=args.root).replay(push, frame_size=args.size).frames

    s = args.size
    frames = []
    for k in range(start, len(ep.frames)):
        canvas = Image.new("RGB", (3 * s + 8, s + 24), "white")
        clip = Image.fromarray(ep.frames[k]).resize((s, s), Image.BILINEAR)
        x, y = ep.pusher_pos[k] * s / ep.workspace
        ImageDraw.Draw(clip).ellipse([x - 6, y - 6, x + 6, y + 6], fill=PALM, outline="white", width=2)
        canvas.paste(clip, (0, 24))
        if k in guess:
            canvas.paste(Image.fromarray(guess[k]).resize((s, s), Image.BILINEAR), (s + 4, 24))
        else:
            ImageDraw.Draw(canvas).text((1.5 * s + 4, s / 2 + 24), "needs 1.5 s of clip first", fill="grey",
                                        anchor="mm", font=font)
        canvas.paste(Image.fromarray(panda[k - start]).resize((s, s), Image.BILINEAR), (2 * s + 8, 24))
        draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(["my clip (palm green)", "predicted 1 s ahead*", "Panda replays my palm"]):
            draw.text((j * (s + 4) + s / 2, 12), label, fill="black", anchor="mm", font=font)
        frames.append(canvas)
    # Each frame gets its own 255 colours plus the palm's: the usual per-frame palette merges the dot, only a few
    # pixels, into the colours around it (as in scripts/panda_replay.py).
    images = []
    for canvas in frames:
        holder = Image.new("P", (1, 1))
        holder.putpalette(canvas.quantize(255).getpalette()[: 3 * 255] + list(PALM))
        images.append(canvas.quantize(palette=holder, dither=Image.Dither.NONE))
    step = ep.timestamps[1] - ep.timestamps[0]
    images[0].save(out / f"{name}.gif", save_all=True, append_images=images[1:], duration=1000 * step, loop=0)
    print(f"{name}: {len(frames)} frames, predictions from frame {first_k}, replay from frame {start} -> "
          f"{out / f'{name}.gif'}")
print("* the nearest real frame (from the training clips) to the predicted embedding")
