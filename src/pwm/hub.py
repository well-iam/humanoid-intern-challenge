"""The public Hugging Face dataset: which files of the run folder it holds, and the files made for it.

It holds my clips (720p H.264 copies of the takes, and the episodes: tracks plus the straightened frames), the
tracks as one table, the held-out split, and for both datasets the cached embeddings, trained predictors,
results and replays that the figures need (but not the V-JEPA 2 embeddings: V-JEPA 2 lost to DINOv2, and its
embeddings are only needed to retrain it). It never holds the test takes (setup checks only) or Push-T itself
(the notebook fetches Push-T from its original URL).
"""

import csv
import fnmatch
import json
from pathlib import Path

from pwm.episodes import Episode

PUBLIC = [  # paths relative to the run root
    "clips/episodes/*.npz",
    "clips/dinov2/**",
    "clips/panda/*",
    "clips/dinov2-30clips/*",  # the first run, 30 clips (its embeddings are the same files as clips/dinov2/)
    "clips/panda-30clips/*",
    "clips/gif-strip/*",
    "clips/tracking/*",
    "clips/logs/train-*.log",
    "clips/aruco-a4.pdf",
    "pusht/dinov2/**",
    "pusht/vjepa2/**",
    "pusht/winner.json",
    "pusht/error-vs-horizon.png",
    "pusht/frame-strip.png",
    "pusht/judge/*",
    "pusht/panda/*",
    "pusht/logs/train-*.log",
    "pusht/logs/seeds/train-*.log",
]
NEVER = [
    "*.partial.npy",
    "pusht/vjepa2/embeddings/*",  # 9.3 GB, only needed to retrain the losing encoder
]
VIDEO_HEIGHT = 720


def public_files(root: Path) -> dict[str, Path]:
    """{path in the dataset: file in the run folder} for every run-folder file the dataset holds."""
    files = {}
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        name = path.relative_to(root).as_posix()
        if any(_match(name, pattern) for pattern in PUBLIC) and not any(fnmatch.fnmatch(name, p) for p in NEVER):
            files[name] = path
    return files


def public_videos(root: Path) -> dict[str, Path]:
    """{path in the dataset: take} for my takes, which go in as 720p copies."""
    return {f"clips/videos/{take.name}": take for take in sorted((Path(root) / "clips" / "raw").glob("*.mp4"))}


def _match(name: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return name.startswith(pattern[:-2])
    return fnmatch.fnmatch(name, pattern) and name.count("/") == pattern.count("/")


def write_tracks(episodes: list[Episode], path: Path) -> None:
    """One row per frame of every clip: time, palm and object position (cm, table frame), object angle, what
    was seen, and the clip's split and kind."""
    columns = ["clip", "split", "kind", "frame", "t_s", "palm_x_cm", "palm_y_cm", "palm_seen", "object_x_cm",
               "object_y_cm", "object_angle_rad", "object_seen", "hand_dir_x", "hand_dir_y"]
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(columns)
        for ep in episodes:
            for t in range(len(ep.timestamps)):
                out.writerow([ep.name, ep.split, ep.kind, t, f"{ep.timestamps[t]:.3f}",
                              *(f"{v:.3f}" for v in ep.pusher_pos[t]), int(ep.pusher_seen[t]),
                              *(f"{v:.3f}" for v in ep.object_pos[t]), f"{ep.object_angle[t]:.4f}",
                              int(ep.object_seen[t]), *(f"{v:.4f}" for v in ep.hand_direction[t])])


def write_split(episodes: list[Episode], path: Path) -> None:
    """The held-out split and each clip's kind."""
    split = {s: [ep.name for ep in episodes if ep.split == s] for s in ("train", "held-out")}
    kinds = {ep.name: ep.kind for ep in episodes}
    path.write_text(json.dumps({**split, "kind": kinds}, indent=1))


def transcode(src: Path, dst: Path, height: int = VIDEO_HEIGHT) -> None:
    """A copy of the take at `height` lines, H.264 (libx264, CRF 20) with its audio (AAC): the claps cut it
    into clips."""
    import av

    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        vin.thread_type = "AUTO"
        width = int(round(vin.width * height / vin.height / 2)) * 2
        vout = out.add_stream("libx264", rate=vin.average_rate)
        vout.width, vout.height, vout.pix_fmt = width, height, "yuv420p"
        vout.options = {"crf": "20", "preset": "medium"}
        vout.time_base = vout.codec_context.time_base = vin.time_base  # keep each frame's time: the rate varies
        streams = [vin]
        if inp.streams.audio:
            ain = inp.streams.audio[0]
            aout = out.add_stream("aac", rate=ain.rate)
            streams.append(ain)
        for packet in inp.demux(*streams):
            for frame in packet.decode():
                if packet.stream.type == "video":
                    small = frame.reformat(width=width, height=height, format="yuv420p")
                    small.pts, small.time_base = frame.pts, frame.time_base
                    out.mux(vout.encode(small))
                else:  # each sound keeps its time against the frames: the claps cut the take
                    out.mux(aout.encode(frame))
        out.mux(vout.encode(None))
        if inp.streams.audio:
            out.mux(aout.encode(None))
