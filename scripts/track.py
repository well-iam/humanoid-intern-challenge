"""Takes -> tracked episodes, with a check report, overlay GIFs and a static-baseline report.

Every take (*.mp4) in the dataset's raw folder (Friday's takes: <root>/clips/raw/) is cut into clips, and the
palm and the object are tracked in each. All takes are in one setup, SETUP (camera height, object, square; see
pwm.datasets.setups). The episodes go to <root>/<dataset>/episodes/ and the checks to
<root>/<dataset>/tracking/: report.md and one <take>.gif per take (palm and object drawn on the straightened
frames). Then the toy encoder gives a static-baseline report in <root>/<dataset>/toy/.

    uv run python scripts/track.py                              # Friday's takes
    uv run python scripts/track.py --dataset clips-test-take    # the test take (setup check only)
"""

import argparse
import traceback

import numpy as np
from PIL import Image, ImageDraw

from pwm import stages
from pwm.datasets import clips, setups, tracking
from pwm.evaluator import STATIC

SETUP = setups.CARPET  # the real takes: the white box on the carpet
GIF_SIZE = 256
SEEN, NOT_SEEN = (40, 220, 40), (150, 150, 150)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", default="clips")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()
root = stages.data_root(args.root)
raw = stages.raw_folder(args.dataset, root)
out = root / args.dataset / "tracking"
out.mkdir(parents=True, exist_ok=True)
videos = sorted(raw.glob("*.mp4"))
assert videos, f"no takes (*.mp4) in {raw}"


def overlay(clip: tracking.TrackedClip, path) -> None:
    """The clip's straightened frames with the object's footprint and the palm drawn on, frames after the hand
    left dimmed. Grey marks: not seen in that frame (held or filled)."""
    scale = GIF_SIZE / clips.WORKSPACE
    corners = tracking.object_corners(clip.object_pos, clip.object_angle, SETUP.object_size) * scale
    frames = []
    for t, image in enumerate(clip.frames):
        im = Image.fromarray(image).resize((GIF_SIZE, GIF_SIZE), Image.LANCZOS)
        if t >= clip.kept:
            im = Image.blend(im, Image.new("RGB", im.size), 0.6)
        d = ImageDraw.Draw(im)
        d.polygon([tuple(p) for p in corners[t]], outline=(255, 210, 0) if clip.object_seen[t] else NOT_SEEN, width=2)
        if t < clip.kept:
            x, y = clip.palm[t] * scale
            dx, dy = clip.hand_direction[t] * 5 * scale  # 5 cm
            colour = SEEN if clip.palm_seen[t] else NOT_SEEN
            d.line([(x, y), (x + dx, y + dy)], fill=colour, width=2)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour)
        label = f"{clip.name.rsplit('-', 1)[1]} {clip.kind} {clip.timestamps[t]:.2f} s" + ("  (cut)" if t >= clip.kept else "")
        d.rectangle([0, 0, GIF_SIZE, 13], fill=(0, 0, 0))
        d.text((3, 1), label, fill=(255, 255, 255))
        frames.append(im)
    path.extend(frames)


tracked, failed = [], []
for video in videos:
    print(f"Tracking {video.name} ...", flush=True)
    try:
        take = tracking.track_take(video, models=root / "models", setup=SETUP)
    except Exception as error:  # one bad take must not stop the others
        traceback.print_exc()
        failed.append((video.name, str(error)))
        continue
    tracked += take
    frames = []
    for clip in take:
        overlay(clip, frames)
    frames[0].save(out / f"{video.stem}.gif", save_all=True, append_images=frames[1:], duration=1000 // clips.FPS,
                   loop=0)

episodes = tracking.to_episodes(tracked)
stages.write_episodes(args.dataset, episodes, root)
split = {ep.name: ep.split for ep in episodes}

# The report.
usable = [t for t in tracked if t.problem is None]
kinds = [t.kind for t in usable]
sheets = np.array([t.sheet for t in tracked])
sizes = np.array([t.object_size for t in usable])
width, length, height = SETUP.object_size


def cm(value: float) -> str:
    """A measured side, or "cut" where the frame's edge cuts it."""
    return "cut" if np.isnan(value) else f"{value:.1f}"


lines = [
    f"# Tracking: {args.dataset}",
    "",
    f"{len(videos)} takes in `{raw}`, {len(tracked)} clips, {len(episodes)} episodes "
    f"({kinds.count('push')} push, {kinds.count('near-miss')} near-miss; "
    f"{sum(s == 'held-out' for s in split.values())} held out) in `{root / args.dataset / 'episodes'}`.",
    "",
]
if failed:
    lines += ["**Takes that failed:**", "", *[f"- {name}: {error}" for name, error in failed], ""]
if len(tracked):
    error = sheets - tracking.SHEET
    measured = [
        f"{name} side {np.nanmean(sheets[:, k]):.2f} cm (error {np.nanmean(error[:, k]):+.2f}, worst "
        f"{np.nanmax(np.abs(error[:, k])):.2f}, in {np.isfinite(sheets[:, k]).sum()} clips)"
        for k, name in enumerate(["long", "short"]) if np.isfinite(sheets[:, k]).any()
    ]
    lines += [
        "## Pixel-to-cm check: the A4 sheet",
        "",
        f"The sheet under the marker, {tracking.SHEET[0]} x {tracking.SHEET[1]} cm, measured in each clip's "
        f"hand-free second (a side the frame's edge cuts is left out): {'; '.join(measured) or 'cut in every clip'}.",
        "",
    ]
if len(usable):
    error = np.abs(sizes - [width, length])
    lines += [
        "## Pixel-to-cm check: the object's top face",
        "",
        f"The object is rigid, {width:g} x {length:g} x {height:g} cm by ruler. Its top face (width x length, after "
        f"the {height:g} cm height fix), median per clip: {sizes[:, 0].mean():.2f} x {sizes[:, 1].mean():.2f} cm on "
        f"average, worst error {error.max():.2f} cm, spread across clips {np.ptp(sizes[:, 0]):.2f} x "
        f"{np.ptp(sizes[:, 1]):.2f} cm.",
        "",
    ]
lines += [
    "## Clips",
    "",
    "Times are seconds after the clap. Hand in / out: first and last frame MediaPipe finds the hand (the episode "
    "ends at the last). Palm seen: clip frames where the hand was found. Moved: the object's start-to-end "
    f"distance (push above {tracking.MOVED:g} cm). Start pose: the object's centre (cm) and angle (degrees). "
    f"\"Hand found late\": less than {tracking.LATE_HAND:g} s before the object starts moving, so the palm path "
    "has little or nothing before contact.",
    "",
    "| clip | kind | split | moved (cm) | hand in | hand out | frames kept | palm seen | object seen "
    "| top face (cm) | start pose | A4 sheet (cm) | flags |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
]
for t in tracked:
    sheet = f"{cm(t.sheet[0])} x {cm(t.sheet[1])}"
    if t.problem:
        lines.append(f"| {t.name} | - | - | - | - | - | - | - | - | - | - | {sheet} | {'; '.join(t.flags)} |")
        continue
    k = t.kept
    x, y = t.object_pos[0]
    lines.append(
        f"| {t.name} | {t.kind} | {split[t.name]} | {t.moved:.1f} | {t.entered:.2f} | {t.left:.2f} | {k} "
        f"| {t.palm_seen[:k].sum()}/{k} | {t.object_seen[:k].sum()}/{k} "
        f"| {t.object_size[0]:.1f} x {t.object_size[1]:.1f} | ({x:.1f}, {y:.1f}), {np.degrees(t.object_angle[0]):.0f} "
        f"| {sheet} | {'; '.join(t.flags) or '-'} |"
    )

# The static baseline through the toy encoder: the episodes flow through the chain.
train = sum(ep.split == "train" for ep in episodes)
lines += ["", "## Static baseline, toy encoder", ""]
try:
    stages.embed(args.dataset, "toy", root)
    report = stages.evaluate(args.dataset, "toy", root)
    lines += [
        f"Held-out windows: {report.n_windows}; probe error on held-out real frames {report.probe_error:.2f} cm.",
        "",
        "| horizon (steps of 0.25 s) | " + " | ".join(str(h) for h in report.horizons) + " |",
        "|---|" + "---:|" * len(report.horizons),
        "| moving windows | " + " | ".join(str(n) for n in report.n_moving) + " |",
        "| static baseline, moving (cm) | " + " | ".join(f"{e:.2f}" for e in report.errors[STATIC]["moving"]) + " |",
        "| static baseline, all (cm) | " + " | ".join(f"{e:.2f}" for e in report.errors[STATIC]["all"]) + " |",
        "",
        f"Full report: `{root / args.dataset / 'toy' / 'report.md'}`.",
    ]
except ValueError as error:  # e.g. the probe needs 5 training clips; the test take has 2
    lines.append(f"Skipped: {error} ({train} training clips here).")

(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nWrote {out / 'report.md'} and {len(videos) - len(failed)} GIFs in {out}")
