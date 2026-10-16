"""Clip adapter, part 2: a take's clips -> tracked clips -> episodes.

- Palm: MediaPipe's HandLandmarker (Tasks API, VIDEO mode, one hand) on every video frame from the clap to
  the clip's end, straightened to a view of the table that reaches 10 cm below the square, where the hand
  comes in: on the square alone the hand is found only once the palm is well inside. The palm position is the
  mean of landmarks 0, 5, 9, 13 and 17. Gaps up to MAX_GAP are filled (longer ones too, but flagged), the path
  inside the hand's visit is smoothed (Savitzky-Golay), and the clip's frames take it at their video frames.
  Before the hand comes in, the palm is held at its first tracked point and flagged as not seen. The hand's
  direction (wrist, landmark 0, to the middle finger's base, landmark 9) is kept next to it.
- Object: the wallet, found in each straightened frame as a dark, not orange (wood and its shadows are
  orange) region of about the wallet's seen size. Its pose is the minimum-area rectangle around it. Frames
  where the hand hides part of it (area below OCCLUDED of the clip's median) are dropped and filled.
- Height fix: the camera sees the palm 1.3 cm and the wallet's top 2 cm above the table, so they look shifted
  away from the point under the lens and slightly too big. Points are moved back toward that point (taken as
  the image centre) by the ratio of heights: true = nadir + (seen - nadir) * (CAMERA_HEIGHT - h) / CAMERA_HEIGHT.
- Conversion check: the A4 sheet under the marker is measured in every clip; it lies flat on the table, so
  its size checks the whole pixel-to-cm conversion.
- Episodes: each clip ends at the last frame where the hand is found (the hand left the view). It is a push if
  the wallet ends more than MOVED from where it started, else a near-miss. HELD_OUT clips of each kind (at most
  half of that kind) are held out, drawn by a fixed seed over all takes together.
"""

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from pwm.datasets import clips
from pwm.episodes import Episode

CAMERA_HEIGHT = 77.0  # cm, lens above the table (ticket 10; it was 73 cm before 16 Oct)
SHEET = (29.7, 21.0)  # cm, A4: the sheet the marker is printed on

# The wallet as the camera sees it, in table cm: (width, length, height), the length along table y at angle 0.
# A ruler gives 11.5 x 10 x 2 cm, but the wallet is not rigid and its sides bulge, so its seen footprint is
# about 13 x 11 cm (decided 16 Oct). Replays use this size too.
WALLET = (11.0, 13.0, 2.0)
DARK = 100  # a wallet pixel has every channel below this...
WARM = 25  # ...and red at most this much above blue (the wood and shadows on it are orange)
SHAPE_TOLERANCE = 2.0  # cm: a dark region is the wallet if both its sides are this close to the wallet's
OCCLUDED = 0.6  # the wallet is partly hidden when its area is below this fraction of the clip's median

HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
PALM = [0, 5, 9, 13, 17]  # MediaPipe hand landmarks: the wrist and the four finger bases
WRIST, MIDDLE_BASE = 0, 9
PALM_HEIGHT = 1.3  # cm: the palm's joints in the pushing pose (the knuckle tops are 1.8 cm)
VIEW = (-5.0, -5.0, 60.0, 70.0)  # the tracking view in table cm: x, y of its top-left corner, width, height
VIEW_SCALE = 8.0  # px per cm in the tracking view
MAX_GAP = 0.25  # seconds: longer gaps in the palm path are flagged
SMOOTHING = (9, 2)  # Savitzky-Golay window (video frames, 0.3 s at 30 fps) and polynomial order

MOVED = 1.0  # cm: a clip is a push if the wallet ends further than this from where it started
HELD_OUT = {"push": 6, "near-miss": 2}  # held-out clips per kind (at most half of each kind)
SEED = 0
STEP = 0.25  # seconds: every 2nd frame at 8 fps, two interleaved sequences
MOVE_THRESHOLD = 1.0  # cm


@dataclass
class TrackedClip:
    """One clip, tracked: per frame (T = the clip's frames, before the end is trimmed) and per clip."""

    name: str
    frames: np.ndarray  # (T, S, S, 3) uint8, the straightened table square
    timestamps: np.ndarray  # (T,) seconds since the clap
    sheet: np.ndarray  # (2,) the A4 sheet's long and short side in table cm, measured in the hand-free second
    drift_px: float  # how far the marker moved during the clip, image px (over 1 px: the camera moved)
    object_pos: np.ndarray  # (T, 2) the wallet's centre on the table, cm (filled; NaN if never found)
    object_angle: np.ndarray  # (T,) radians, see pwm.replay (0: long side along table y); not wrapped
    object_seen: np.ndarray  # (T,) bool: False where it was not found or partly hidden (then filled)
    object_size: np.ndarray  # (2,) the seen footprint (width, length), cm: the median over the frames it is seen
    palm: np.ndarray  # (T, 2) the palm position on the table, cm (NaN if the hand is never found)
    palm_seen: np.ndarray  # (T,) bool: False where the hand was not found in that frame (held or filled)
    hand_direction: np.ndarray  # (T, 2) unit vector from the wrist to the middle finger's base, table frame
    entered: float  # seconds after the clap: the hand is first found (NaN if never)
    left: float  # seconds after the clap: the hand is last found
    kept: int  # frames up to the last one where the hand is found: the episode's length
    palm_gaps: list[tuple[float, float]] = field(default_factory=list)  # (start s, length s) longer than MAX_GAP

    @property
    def problem(self) -> str | None:
        """Why the clip cannot become an episode, or None."""
        if not self.palm_seen.any():
            return "the hand is never found"
        if not self.object_seen.any():
            return "the wallet is never found"
        return None

    @property
    def moved(self) -> float:
        """How far the wallet ends from where it started, cm."""
        return float(np.linalg.norm(self.object_pos[self.kept - 1] - self.object_pos[0]))

    @property
    def kind(self) -> str:
        return "push" if self.moved > MOVED else "near-miss"


def hand_model(models: Path) -> Path:
    """The MediaPipe hand landmarker model (float16, version 1), downloaded into `models` on first use."""
    path = Path(models) / "mediapipe" / "hand_landmarker-float16-v1.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(HAND_MODEL, partial)
        partial.rename(path)
    return path


def track_take(video: str | Path, models: Path) -> list[TrackedClip]:
    """Cut a take into clips and track the palm and the wallet in each."""
    video = Path(video)
    take = clips.cut_take(video)
    stills, hands = _scan(video, take, hand_model(models))
    tracked = []
    for clip, still, hand in zip(take.clips, stills, hands):
        nadir = clip.to_cm(np.array([still.shape[1] / 2, still.shape[0] / 2]))
        picks = np.searchsorted(take.frame_times, clip.source_times)  # the clip's video frames
        tracked.append(
            TrackedClip(
                name=clip.name,
                frames=clip.frames,
                timestamps=clip.timestamps,
                sheet=sheet_size(still, clip),
                drift_px=float(np.nanmax(clip.drift_px)),
                **_track_object(clip.frames, clip.timestamps, nadir),
                **_palm_path(hand, take.frame_times, clip.clap, picks, nadir),
            )
        )
    return tracked


def to_episodes(tracked: list[TrackedClip]) -> list[Episode]:
    """Episodes from tracked clips: each trimmed to the hand leaving, labelled push or near-miss, and split.
    Clips with a problem (see TrackedClip.problem) are left out."""
    usable = sorted((t for t in tracked if t.problem is None), key=lambda t: t.name)
    held_out = _held_out([t.name for t in usable], [t.kind for t in usable])
    episodes = []
    for t in usable:
        keep = slice(0, t.kept)
        episodes.append(
            Episode(
                name=t.name,
                frames=t.frames[keep],
                timestamps=t.timestamps[keep],
                pusher_pos=t.palm[keep],
                object_pos=t.object_pos[keep],
                workspace=clips.WORKSPACE,
                unit="cm",
                split="held-out" if t.name in held_out else "train",
                kind=t.kind,
                pusher_seen=t.palm_seen[keep],
                object_seen=t.object_seen[keep],
                object_angle=t.object_angle[keep],
                hand_direction=t.hand_direction[keep],
            )
        )
    return episodes


def build(raw_dir: Path) -> list[Episode]:
    """Dataset-adapter entry point: every take (*.mp4) in `raw_dir`, cut, tracked and split. The hand model is
    kept in <root>/models, the raw folder being <root>/<dataset>/raw (or <root>/clips/test-take)."""
    raw_dir = Path(raw_dir)
    models = raw_dir.resolve().parents[1] / "models"
    return to_episodes([t for video in sorted(raw_dir.glob("*.mp4")) for t in track_take(video, models)])


def _held_out(names: list[str], kinds: list[str]) -> set[str]:
    """The held-out clips: HELD_OUT of each kind (at most half of that kind), drawn by a fixed seed."""
    rng = np.random.default_rng(SEED)
    held_out = set()
    for kind, wanted in HELD_OUT.items():
        of_kind = sorted(n for n, k in zip(names, kinds) if k == kind)
        n = min(wanted, len(of_kind) // 2)
        held_out |= {of_kind[i] for i in rng.permutation(len(of_kind))[:n]}
    return held_out


def object_corners(pos: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """The wallet's footprint corners (..., 4, 2) in table cm, for centres (..., 2) and angles (...), using the
    seen size WALLET: its edges, e.g. to find where the hand touches it."""
    pos, angle = np.asarray(pos, dtype=np.float64), np.asarray(angle, dtype=np.float64)
    across = np.stack([np.cos(angle), np.sin(angle)], axis=-1)[..., None, :] * WALLET[0] / 2
    along = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)[..., None, :] * WALLET[1] / 2
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    return pos[..., None, :] + signs[:, :1] * across + signs[:, 1:] * along


def lift(points: np.ndarray, nadir: np.ndarray, height: float) -> np.ndarray:
    """Seen table cm -> true table cm of points `height` cm above the table (the height fix)."""
    return nadir + (np.asarray(points) - nadir) * (CAMERA_HEIGHT - height) / CAMERA_HEIGHT


def _scan(video: Path, take: clips.Take, model: Path):
    """One pass over the take's video: for each clip, a hand-free raw frame (for the sheet and the image size)
    and the hand landmarks in table cm (as seen, before the height fix) per video frame where a hand is found,
    from the end of the clap to the clip's last frame."""
    import mediapipe as mp  # here, not at the top: it takes a second to load, and only tracking needs it
    from mediapipe.tasks.python import BaseOptions, vision

    x0, y0, width, height = VIEW
    table_to_view = np.array([[VIEW_SCALE, 0, -x0 * VIEW_SCALE], [0, VIEW_SCALE, -y0 * VIEW_SCALE], [0, 0, 1]])
    size = (round(width * VIEW_SCALE), round(height * VIEW_SCALE))
    times = take.frame_times
    spans = [
        np.flatnonzero((times >= clip.clap + clips.HAND_FREE[0]) & (times <= clip.source_times[-1]))
        for clip in take.clips
    ]
    still_at = [int(clips._nearest(times, np.array([clip.clap + np.mean(clips.HAND_FREE)]))[0]) for clip in take.clips]
    owners = {}
    for k, span in enumerate(spans):
        for i in span.tolist():
            owners.setdefault(i, []).append(k)

    stills, hands, open_ = [None] * len(spans), [{} for _ in spans], {}
    options = dict(base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO,
                   num_hands=1)
    for i, image in clips._frames(video, times, set(owners), "rgb24"):
        for k in owners[i]:
            clip = take.clips[k]
            if i == still_at[k]:
                stills[k] = image
            if k not in open_:
                open_[k] = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(**options))
            view = cv2.warpPerspective(image, table_to_view @ clip.image_to_table, size, flags=cv2.INTER_LINEAR)
            found = open_[k].detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(view)),
                int(round((times[i] - clip.clap) * 1000)),
            )
            if found.hand_landmarks:
                xy = np.array([[p.x, p.y] for p in found.hand_landmarks[0]])
                hands[k][i] = (x0, y0) + xy * (width, height)
            if i == spans[k][-1]:
                open_.pop(k).close()
    for landmarker in open_.values():
        landmarker.close()
    return stills, hands


def _palm_path(hand: dict[int, np.ndarray], times: np.ndarray, clap: float, picks: np.ndarray, nadir: np.ndarray):
    """The palm path at the clip's frames (video frames `picks`) from the landmarks found per video frame."""
    t = len(picks)
    if not hand:
        nan = np.full((t, 2), np.nan)
        return dict(palm=nan, palm_seen=np.zeros(t, bool), hand_direction=nan, entered=np.nan, left=np.nan, kept=0)
    found = np.array(sorted(hand))
    landmarks = lift(np.stack([hand[i] for i in found]), nadir, PALM_HEIGHT)
    palm = landmarks[:, PALM].mean(axis=1)
    direction = landmarks[:, MIDDLE_BASE] - landmarks[:, WRIST]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    # Every video frame of the hand's visit, gaps filled in time, then smoothed. Outside the visit: held.
    visit = np.arange(found[0], found[-1] + 1)
    filled = np.stack([np.interp(times[visit], times[found], v) for v in (*palm.T, *direction.T)], axis=1)
    window, order = SMOOTHING
    if len(visit) >= window:
        filled = savgol_filter(filled, window, order, axis=0, mode="interp")
    at = np.clip(picks, visit[0], visit[-1]) - visit[0]
    path = filled[at]
    direction = path[:, 2:] / np.linalg.norm(path[:, 2:], axis=1, keepdims=True)

    gaps = np.diff(times[found])
    long = np.flatnonzero(gaps > MAX_GAP)
    return dict(
        palm=path[:, :2],
        palm_seen=np.isin(picks, found),
        hand_direction=direction,
        entered=float(times[found[0]] - clap),
        left=float(times[found[-1]] - clap),
        kept=int(np.sum(picks <= found[-1])),
        palm_gaps=[(float(times[found[g]] - clap), float(gaps[g])) for g in long],
    )


def _track_object(frames: np.ndarray, timestamps: np.ndarray, nadir: np.ndarray):
    """The wallet in each straightened frame: its centre and angle after the height fix, filled in time where it
    is not seen (not found, or partly hidden by the hand), and its seen footprint."""
    n = len(frames)
    pos, angle, size, area = np.full((n, 2), np.nan), np.full(n, np.nan), np.full((n, 2), np.nan), np.zeros(n)
    for t, frame in enumerate(frames):
        found = _find_object(frame)
        if found is not None:
            corners, area[t] = found
            pos[t], angle[t], size[t] = _pose(lift(corners, nadir, WALLET[2]))
    seen = np.isfinite(angle)
    if seen.any():
        seen &= area >= OCCLUDED * np.median(area[seen])
        turn = np.unwrap(2 * angle[seen]) / 2  # continuous over the clip (a rectangle repeats every half turn)
        pos = np.stack([np.interp(timestamps, timestamps[seen], v) for v in pos[seen].T], axis=1)
        angle = np.interp(timestamps, timestamps[seen], turn)
    return dict(
        object_pos=pos,
        object_angle=angle,
        object_seen=seen,
        object_size=np.median(size[seen], axis=0) if seen.any() else np.full(2, np.nan),
    )


def _find_object(frame: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The wallet's seen outline in a straightened frame: the corners (4, 2) of the minimum-area rectangle around
    the largest dark, not orange region of about its size, in table cm, and that region's area in px; None if
    there is none."""
    rgb = frame.astype(np.int16)
    mask = ((rgb.max(axis=-1) < DARK) & (rgb[..., 0] - rgb[..., 2] < WARM)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    px = clips.SIZE / clips.WORKSPACE  # px per cm
    for label in 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]:
        if stats[label, cv2.CC_STAT_AREA] < 0.25 * WALLET[0] * WALLET[1] * px**2:
            break
        ys, xs = np.nonzero(labels == label)
        centre, (w, h), angle = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        # The rectangle runs through pixel centres: the region's outline is one pixel wider each way.
        rect = ((centre[0] + 0.5, centre[1] + 0.5), (w + 1, h + 1), angle)
        sides = np.sort([w + 1, h + 1]) / px
        if np.all(np.abs(sides - np.sort(WALLET[:2])) < SHAPE_TOLERANCE):
            return cv2.boxPoints(rect).astype(np.float64) / px, float(stats[label, cv2.CC_STAT_AREA])
    return None


def _pose(corners: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Centre, angle and (width, length) of a rectangle given by its corners in order. The angle turns table x
    toward table y (see pwm.replay); at angle 0 the long side lies along table y. It is in (-pi/2, pi/2]."""
    sides = np.diff(np.concatenate([corners, corners[:1]]), axis=0)[:2]
    lengths = np.linalg.norm(sides, axis=1)
    along = sides[np.argmax(lengths)]  # the long side: (-sin a, cos a) * length
    angle = np.arctan2(-along[0], along[1])
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2  # a rectangle looks the same turned by half a turn
    return corners.mean(axis=0), float(angle), np.sort(lengths)


def sheet_size(image: np.ndarray, clip: clips.Clip) -> np.ndarray:
    """The A4 sheet's long and short side in table cm, in a raw video frame where no hand covers it.

    The sheet is the large pale, unsaturated region around the marker (the wood is orange); its outline is the
    convex hull, so the marker's black squares inside it do not matter."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pale = ((hsv[..., 1] < 45) & (hsv[..., 2] > 120)).astype(np.uint8)
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    _, labels, _, _ = cv2.connectedComponentsWithStats(pale)
    # The sheet: the pale region just outside the marker's black border (0.7 cm out; the sheet's margin is 1.5 cm).
    centre = clip.corners.mean(axis=0)
    ring = centre + (clip.corners - centre) * (1 + 1.4 / clips.MARKER_SIDE)
    ring = np.concatenate([np.linspace(a, b, 50) for a, b in zip(ring, np.roll(ring, -1, axis=0))])
    around = labels[np.round(ring[:, 1]).astype(int), np.round(ring[:, 0]).astype(int)]
    sheet = np.bincount(around[around > 0]).argmax()
    ys, xs = np.nonzero(labels == sheet)
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.float32)).reshape(-1, 2)
    _, (w, h), _ = cv2.minAreaRect(clip.to_cm(hull.astype(np.float64)).astype(np.float32))
    return np.array([max(w, h), min(w, h)])
