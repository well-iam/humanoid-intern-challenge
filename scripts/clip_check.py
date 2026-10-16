"""Check a take: claps, frame picking, the marker, camera drift, the glare, and the straightened view.

Cuts the take into clips at the claps and writes into <video folder>/check/<take>/:
report.md, straightened.gif (every clip's straightened frames at 8 fps) and layout.png (the first frame
of the first clip with the table square, the marker and any glare drawn on it).
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

from pwm.datasets import clips

GLARE = 245  # a pixel is glare when all three channels are at least this bright (the table never is)
GIF_SIZE = 256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path)
parser.add_argument("--out", type=Path, help="output folder (default: <video folder>/check/<take>)")
args = parser.parse_args()
out = args.out or args.video.parent / "check" / args.video.stem
out.mkdir(parents=True, exist_ok=True)

take = clips.cut_take(args.video)

# The raw video frame each clip starts on: for the glare and the layout picture.
starts = {}
with av.open(str(args.video)) as container:
    wanted = [clip.source_times[0] for clip in take.clips]
    for frame in container.decode(video=0):
        for k, t in enumerate(wanted):
            if k not in starts and abs(frame.time - t) < 1e-3:
                starts[k] = frame.to_ndarray(format="rgb24")
        if len(starts) == len(wanted):
            break


def glare(clip, raw):
    """Glare pixels of the raw frame: how many, how many inside the square, and the closest one's distance
    to the square in cm (0 if inside)."""
    ys, xs = np.nonzero(raw.min(axis=-1) >= GLARE)
    if len(xs) == 0:
        return 0, 0, np.nan
    cm = clip.to_cm(np.stack([xs, ys], axis=1).astype(float))
    outside = np.maximum(np.maximum(-cm, cm - clips.WORKSPACE), 0)
    distance = np.hypot(outside[:, 0], outside[:, 1])
    return len(xs), int((distance == 0).sum()), float(distance.min())


frame_gaps = np.diff(take.frame_times) * 1000
lines = [
    f"# Clip check: {take.name}",
    "",
    f"Video: {len(take.frame_times)} frames over {take.frame_times[-1]:.1f} s, "
    f"{(len(take.frame_times) - 1) / (take.frame_times[-1] - take.frame_times[0]):.2f} fps on average, "
    f"frame gaps {frame_gaps.min():.1f}-{frame_gaps.max():.1f} ms.",
    "",
    "## Claps",
    "",
    f"{len(take.claps)} claps (loudness above the take's noise floor; a clap must be within "
    f"{clips.CLAP_RANGE:.0f} dB of the loudest and {clips.CLAP_GAP:.0f} s from any louder one):",
    "",
    "| clap | time (s) | loudness (dB) |",
    "|---:|---:|---:|",
    *[f"| {k + 1} | {t:.2f} | {db:.1f} |" for k, (t, db) in enumerate(zip(take.claps, take.clap_db))],
    "",
    f"Loudest other sound: {take.other[1]:.1f} dB at {take.other[0]:.2f} s "
    f"(cut-off {take.clap_db.max() - clips.CLAP_RANGE:.1f} dB).",
    "",
    "## Clips",
    "",
    f"Frames picked at {clips.FPS} fps by timestamp; a clip lasts at most {clips.MAX_LENGTH:.0f} s. "
    f"Marker drift: the largest corner distance from the clip's own reference (the mean over its hand-free "
    f"second, {clips.HAND_FREE[0]}-{clips.HAND_FREE[1]} s after the clap). Flags at {clips.MAX_DRIFT:.0f} px.",
    "",
    "| clip | clap (s) | frames | length (s) | worst frame-time gap (ms) | marker, hand-free second "
    "| marker, clip frames | drift max (px / cm) | step from previous clip (px / cm) | px per cm | flags |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for clip in take.clips:
    gap = np.abs(clip.source_times - clip.clap - clip.timestamps).max() * 1000
    found = np.isfinite(clip.drift_px)
    drift = np.nanmax(clip.drift_px)
    flags = [name for name, on in [("drifted", clip.drifted), ("camera moved", clip.camera_moved)] if on]
    step = "-" if np.isnan(clip.step_px) else f"{clip.step_px:.2f} / {clip.step_px / clip.px_per_cm:.3f}"
    lines.append(
        f"| {clip.name} | {clip.clap:.2f} | {len(clip.frames)} | {clip.timestamps[-1] + 1 / clips.FPS:.2f} "
        f"| {gap:.1f} | {clip.hand_free_found}/{clip.hand_free_frames} | {found.sum()}/{len(found)} "
        f"| {drift:.2f} / {drift / clip.px_per_cm:.3f} | {step} | {clip.px_per_cm:.2f} | {', '.join(flags) or '-'} |"
    )

found = sum(c.hand_free_found + np.isfinite(c.drift_px).sum() for c in take.clips)
total = sum(c.hand_free_frames + len(c.drift_px) for c in take.clips)
lines += ["", f"Marker detection rate over every frame looked at: {found}/{total} ({found / total:.1%}).", ""]

lines += [
    "## Glare",
    "",
    f"Glare pixels: all channels at least {GLARE}, in the raw frame each clip starts on.",
    "",
    "| clip | glare pixels | inside the square | closest to the square (cm) | glare in the straightened frames |",
    "|---|---:|---:|---:|---:|",
]
for k, clip in enumerate(take.clips):
    n, inside, distance = glare(clip, starts[k])
    straightened = (clip.frames.min(axis=-1) >= GLARE).mean()
    lines.append(f"| {clip.name} | {n} | {inside} | {distance:.2f} | {straightened:.4%} |")
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Layout picture: the first clip's first frame, the square, the marker and the glare.
clip, raw = take.clips[0], starts[0].copy()
table_to_image = np.linalg.inv(clip.image_to_table)
square = cv2.perspectiveTransform((clips.WORKSPACE * clips.UNIT_SQUARE).reshape(-1, 1, 2), table_to_image)
raw[raw.min(axis=-1) >= GLARE] = (255, 0, 255)
cv2.polylines(raw, [np.round(square).astype(np.int32)], True, (0, 220, 0), 4)
cv2.polylines(raw, [np.round(clip.corners).astype(np.int32)], True, (255, 0, 0), 4)
Image.fromarray(raw).resize((raw.shape[1] // 2, raw.shape[0] // 2), Image.LANCZOS).save(out / "layout.png")

# GIF: every clip's straightened frames, labelled.
frames = []
for clip in take.clips:
    for t, image in zip(clip.timestamps, clip.frames):
        small = Image.fromarray(image).resize((GIF_SIZE, GIF_SIZE), Image.LANCZOS)
        ImageDraw.Draw(small).text((4, 2), f"{clip.name.rsplit('-', 1)[1]}  {t:.2f} s", fill=(255, 255, 255))
        frames.append(small)
frames[0].save(out / "straightened.gif", save_all=True, append_images=frames[1:], duration=1000 // clips.FPS, loop=0)

print((out / "report.md").read_text(encoding="utf-8"))
print(f"Wrote {out / 'report.md'}, {out / 'straightened.gif'} and {out / 'layout.png'}")
