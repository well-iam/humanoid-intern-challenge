"""Clip adapter, part 2: a take's clips -> tracked clips -> episodes.

- Palm: MediaPipe's HandLandmarker (Tasks API, VIDEO mode, one hand) on every video frame from the clap to
  the clip's end, straightened to a view of the table that reaches 15 cm below the square (or to the frame's
  edge, if nearer), where the hand comes in: on the square alone the hand is found only once the palm is well
  inside. The palm position is the mean of landmarks 0, 5, 9, 13 and 17. Gaps up to MAX_GAP are filled (longer
  ones too, but flagged), the path inside the hand's visit is smoothed (Savitzky-Golay), and the clip's frames
  take it at their video frames. Before the hand comes in, the palm is held at its first tracked point and
  flagged as not seen. The hand's direction (wrist, landmark 0, to the middle finger's base, landmark 9) is
  kept next to it. A hand found less than LATE_HAND before the object starts moving is flagged: its palm path
  has little or nothing before contact.
- Object: found in each straightened frame as the largest region of the setup's object pixels (its top face:
  white for the box on the carpet) of about its size. Its pose is the minimum-area rectangle around it. Frames
  where the hand hides part of it (area below OCCLUDED of the clip's median) are dropped and filled.
- Height fix: the camera sees the palm 1.3 cm and the object's top face its height above the table, so they
  look shifted away from the point under the lens and slightly too big. Points are moved back toward that point
  (taken as the image centre) by the ratio of heights: true = nadir + (seen - nadir) * (camera - h) / camera.
- Setup: the camera height, the object's size and colour and where the square lies come from the take's setup
  (pwm.datasets.setups); by default CARPET, the real takes' setup.
- Conversion check: the A4 sheet under the marker is measured in every clip; it lies flat on the table, so
  its size checks the whole pixel-to-cm conversion.
- Episodes: each clip ends at the last frame where the hand is found (the hand left the view). It is a push if
  the object ends more than MOVED from where it started, else a near-miss. Held out: ticket 13's 8 clips of the
  first takes, plus NEW_HELD_OUT of each kind drawn by a fixed seed over the later takes. EXCLUDED clips (junk
  the clap rule cut, checked by eye) are left out.
"""

import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from pwm.datasets import clips, setups
from pwm.datasets.setups import Setup
from pwm.episodes import Episode

SHEET = (29.7, 21.0)  # cm, A4: the sheet the marker is printed on
SHAPE_TOLERANCE = 2.0  # cm: a region is the object if both its sides are this close to the object's
OCCLUDED = 0.6  # the object is partly hidden when its area is below this fraction of the clip's median

HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
PALM = [0, 5, 9, 13, 17]  # MediaPipe hand landmarks: the wrist and the four finger bases
WRIST, MIDDLE_BASE = 0, 9
PALM_HEIGHT = 1.3  # cm: the palm's joints in the pushing pose (the knuckle tops are 1.8 cm)
VIEW = (-5.0, -5.0, 60.0, 70.0)  # the tracking view in table cm: x, y of its top-left corner, width, height
VIEW_SCALE = 8.0  # px per cm in the tracking view
MAX_GAP = 0.25  # seconds: longer gaps in the palm path are flagged
SMOOTHING = (9, 2)  # Savitzky-Golay window (video frames, 0.3 s at 30 fps) and polynomial order

MOVED = 1.0  # cm: a clip is a push if the object ends further than this from where it started
STARTS_MOVING = 0.5  # cm: the object has started moving once it is this far from where it started
LATE_HAND = 0.25  # seconds: a hand found less than this before the object starts moving is flagged
# The held-out split. The first three takes (14:13-14:16) keep the 8 clips ticket 13 held out (6 pushes, 2
# near-misses), so the 30-clip results stay comparable. Every later take adds NEW_HELD_OUT clips of each kind (at
# most half of that kind), drawn by SEED over the later takes' sorted clip names. Everything else trains.
FIRST_TAKES = ("VID_20261016_141304", "VID_20261016_141455", "VID_20261016_141631")
FIRST_HELD_OUT = (
    "VID_20261016_141304-03", "VID_20261016_141304-05", "VID_20261016_141304-11", "VID_20261016_141455-02",
    "VID_20261016_141455-06", "VID_20261016_141455-09", "VID_20261016_141631-03", "VID_20261016_141631-08",
)
NEW_HELD_OUT = {"push": 4, "near-miss": 6}
SEED = 0
# Clips left out of the episodes: the clap rule cut them, but they are not a push or a near-miss (checked by eye
# on the frames). Clip name -> why.
EXCLUDED: dict[str, str] = {
    "VID_20261016_164109-06": "a push, then the hand comes back within half a second and lifts and turns the "
                              "object to reset it",
    "VID_20261016_164526-04": "a near-miss, then the hand lifts and turns the object to reset it without leaving "
                              "the view (labelled push)",
    "VID_20261016_165020-06": "the clap (33.06 s) came during the previous near-miss: the hand is in view in the "
                              "hand-free second, leaves, and comes back only to reset the object",
    "VID_20261016_165020-08": "between claps 3.1 s apart (43.55 and 46.62 s): only the hand resetting the object "
                              "(labelled push)",
}
# The hand unseen this long, once found, has left the view: the episode ends there. A later return (in the later
# takes, the hand coming back to reset the object inside the 6 s cap) is not part of the push or the near-miss.
LEFT_VIEW = 0.75  # seconds
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
    object_pos: np.ndarray  # (T, 2) the object's centre on the table, cm (filled; NaN if never found)
    object_angle: np.ndarray  # (T,) radians, see pwm.replay (0: long side along table y); not wrapped
    object_seen: np.ndarray  # (T,) bool: False where it was not found or partly hidden (then filled)
    object_size: np.ndarray  # (2,) its top face (width, length) after the height fix, cm: median over the frames seen
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
            return "the object is never found"
        return None

    @property
    def moved(self) -> float:
        """How far the object ends from where it started, cm."""
        return float(np.linalg.norm(self.object_pos[self.kept - 1] - self.object_pos[0]))

    @property
    def kind(self) -> str:
        return "push" if self.moved > MOVED else "near-miss"

    @property
    def starts_moving(self) -> float:
        """Seconds after the clap: the first frame where the object is more than STARTS_MOVING from where it
        started (NaN if it never is)."""
        away = np.linalg.norm(self.object_pos[: self.kept] - self.object_pos[0], axis=1) > STARTS_MOVING
        return float(self.timestamps[np.argmax(away)]) if away.any() else np.nan

    @property
    def flags(self) -> list[str]:
        """What to look at in this clip (the tracking report lists them)."""
        flags = [self.problem] if self.problem else []
        if self.palm_gaps:
            flags.append("palm gaps " + ", ".join(f"{g:.2f} s at {s:.2f}" for s, g in self.palm_gaps))
        if self.problem is None and self.kept == len(self.timestamps):
            flags.append("hand still in view at the 6 s cap")
        if self.drift_px > clips.MAX_DRIFT:
            flags.append(f"camera moved {self.drift_px:.1f} px during the clip")
        if self.entered > self.starts_moving - LATE_HAND:
            flags.append(f"hand found late: at {self.entered:.2f} s, the object moves at {self.starts_moving:.2f} s")
        return flags


def hand_model(models: Path) -> Path:
    """The MediaPipe hand landmarker model (float16, version 1), downloaded into `models` on first use."""
    path = Path(models) / "mediapipe" / "hand_landmarker-float16-v1.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(HAND_MODEL, partial)
        partial.rename(path)
    return path


def track_take(video: str | Path, models: Path, setup: Setup = setups.CARPET) -> list[TrackedClip]:
    """Cut a take into clips and track the palm and the object in each."""
    video = Path(video)
    take = clips.cut_take(video, setup.square)
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
                **_track_object(clip.frames, clip.timestamps, nadir, setup),
                **_palm_path(hand, take.frame_times, clip.clap, picks, nadir, setup.camera_height),
            )
        )
    return tracked


def to_episodes(tracked: list[TrackedClip], excluded: dict[str, str] = EXCLUDED) -> list[Episode]:
    """Episodes from tracked clips: each trimmed to the hand leaving (see first_visit), labelled push or
    near-miss, and split. Clips with a problem (see TrackedClip.problem) and the `excluded` clips are left out,
    before the split."""
    tracked = [first_visit(t) for t in tracked]
    usable = sorted((t for t in tracked if t.problem is None and t.name not in excluded), key=lambda t: t.name)
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


def build(raw_dir: Path, setup: Setup = setups.CARPET) -> list[Episode]:
    """Dataset-adapter entry point: every take (*.mp4) in `raw_dir`, cut, tracked and split. The hand model is
    kept in <root>/models, the raw folder being <root>/<dataset>/raw (or <root>/clips/test-take)."""
    raw_dir = Path(raw_dir)
    models = raw_dir.resolve().parents[1] / "models"
    return to_episodes([t for video in sorted(raw_dir.glob("*.mp4")) for t in track_take(video, models, setup)])


def first_visit(clip: TrackedClip) -> TrackedClip:
    """The clip ending with the hand's first visit: at the last frame before the hand, once found, goes unseen
    for LEFT_VIEW or longer. Shorter gaps (the hand lost for a moment) stay inside the visit."""
    seen = np.flatnonzero(clip.palm_seen[: clip.kept])
    gaps = np.flatnonzero(np.diff(seen) - 1 >= LEFT_VIEW * clips.FPS)
    if len(gaps) == 0:
        return clip
    kept = int(seen[gaps[0]]) + 1
    return replace(clip, kept=kept, left=float(clip.timestamps[kept - 1]))


def _held_out(names: list[str], kinds: list[str]) -> set[str]:
    """The held-out clips: FIRST_HELD_OUT from the first takes, and NEW_HELD_OUT of each kind (at most half of
    that kind) drawn by a fixed seed over the later takes."""
    held_out = set(FIRST_HELD_OUT) & set(names)
    rng = np.random.default_rng(SEED)
    for kind, wanted in NEW_HELD_OUT.items():
        of_kind = sorted(n for n, k in zip(names, kinds) if k == kind and not n.startswith(FIRST_TAKES))
        n = min(wanted, len(of_kind) // 2)
        held_out |= {of_kind[i] for i in rng.permutation(len(of_kind))[:n]}
    return held_out


def object_corners(pos: np.ndarray, angle: np.ndarray, size=setups.CARPET.object_size) -> np.ndarray:
    """The object's footprint corners (..., 4, 2) in table cm, for centres (..., 2), angles (...) and its size
    (width, length, ...; the setup's object_size): its edges, e.g. to find where the hand touches it."""
    pos, angle = np.asarray(pos, dtype=np.float64), np.asarray(angle, dtype=np.float64)
    across = np.stack([np.cos(angle), np.sin(angle)], axis=-1)[..., None, :] * size[0] / 2
    along = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)[..., None, :] * size[1] / 2
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    return pos[..., None, :] + signs[:, :1] * across + signs[:, 1:] * along


def lift(points: np.ndarray, nadir: np.ndarray, height: float, camera: float) -> np.ndarray:
    """Seen table cm -> true table cm of points `height` cm above the table, the camera being `camera` cm above
    it (the height fix)."""
    return nadir + (np.asarray(points) - nadir) * (camera - height) / camera


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


def _palm_path(hand: dict[int, np.ndarray], times: np.ndarray, clap: float, picks: np.ndarray, nadir: np.ndarray,
               camera: float):
    """The palm path at the clip's frames (video frames `picks`) from the landmarks found per video frame, the
    camera `camera` cm above the table."""
    t = len(picks)
    if not hand:
        nan = np.full((t, 2), np.nan)
        return dict(palm=nan, palm_seen=np.zeros(t, bool), hand_direction=nan, entered=np.nan, left=np.nan, kept=0)
    found = np.array(sorted(hand))
    landmarks = lift(np.stack([hand[i] for i in found]), nadir, PALM_HEIGHT, camera)
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


def _track_object(frames: np.ndarray, timestamps: np.ndarray, nadir: np.ndarray, setup: Setup):
    """The object in each straightened frame: its centre and angle after the height fix, filled in time where it
    is not seen (not found, or partly hidden by the hand), and the size of its top face."""
    n = len(frames)
    pos, angle, size, area = np.full((n, 2), np.nan), np.full(n, np.nan), np.full((n, 2), np.nan), np.zeros(n)
    for t, frame in enumerate(frames):
        found = _find_object(frame, setup)
        if found is not None:
            corners, area[t] = found
            pos[t], angle[t], size[t] = _pose(lift(corners, nadir, setup.object_size[2], setup.camera_height))
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


def _find_object(frame: np.ndarray, setup: Setup) -> tuple[np.ndarray, float] | None:
    """The object's top face as seen in a straightened frame: the corners (4, 2) of the minimum-area rectangle
    around the largest region of the setup's object pixels of about its size, in table cm, and that region's area
    in px; None if there is none. The opening drops specks (e.g. the carpet's pale flecks)."""
    mask = setup.object_pixels(frame).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    px = clips.SIZE / clips.WORKSPACE  # px per cm
    width, length = setup.object_size[:2]
    for label in 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]:
        if stats[label, cv2.CC_STAT_AREA] < 0.25 * width * length * px**2:
            break
        ys, xs = np.nonzero(labels == label)
        centre, (w, h), angle = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        # The rectangle runs through pixel centres: the region's outline is one pixel wider each way.
        rect = ((centre[0] + 0.5, centre[1] + 0.5), (w + 1, h + 1), angle)
        sides = np.sort([w + 1, h + 1]) / px
        if np.all(np.abs(sides - np.sort([width, length])) < SHAPE_TOLERANCE):
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
    """The A4 sheet's long and short side in table cm, in a raw video frame where no hand covers it. A side that
    the frame's edge cuts is NaN.

    The sheet is the large pale, unsaturated region around the marker (the wood is orange, the carpet dark; an
    opening drops the carpet's pale flecks). Its outline is the convex hull, so the marker's black squares
    inside it do not matter."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pale = ((hsv[..., 1] < 45) & (hsv[..., 2] > 120)).astype(np.uint8)
    pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
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
    rect = cv2.minAreaRect(clip.to_cm(hull.astype(np.float64)).astype(np.float32))
    edges = np.diff(cv2.boxPoints(rect)[:3], axis=0)  # two adjacent edges: the rectangle's two sides
    sides = np.linalg.norm(edges, axis=1)
    # Cut by the frame's edge: the sheet's pixels on the frame's border all lie at one end of that side.
    border = (xs == 0) | (ys == 0) | (xs == image.shape[1] - 1) | (ys == image.shape[0] - 1)
    cut = np.zeros(2, bool)
    if border.any():
        on_edge = clip.to_cm(np.stack([xs[border], ys[border]], axis=1).astype(np.float64))
        reach = np.abs((on_edge - rect[0]) @ (edges / sides[:, None]).T)  # distance from the centre along each side
        cut = np.median(reach, axis=0) > sides / 2 - 1.0
    order = np.argsort(-sides)  # the long side first
    return np.where(cut, np.nan, sides)[order]
