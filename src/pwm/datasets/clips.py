"""Clip adapter, part 1: a take (one phone video) -> clips cut at the claps, straightened to the table square.

- Claps: the loudest sounds of the take, at least 3 s apart. A clip starts at its clap and lasts at most
  6 s (or until the next clap).
- Frames: picked by timestamp at 8 fps, each the video frame nearest its time, so the phone's uneven frame
  rate does not change the step.
- Marker: the printed ArUco marker is found in every frame used. Each clip's homography (image px -> table
  cm) comes from the marker's corners averaged over the hand-free second after its clap; the other frames
  only check that the camera stayed still (drift) and that it did not move since the previous clip (step).
- Straightening: every clip frame is warped to the 50 x 50 cm table square, SIZE px a side, in the
  camera's orientation. Table cm: origin at the view's top-left corner, x to the right, y down. Where the
  square lies next to the marker depends on the setup (pwm.datasets.setups).
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from pwm.datasets import setups

WORKSPACE = 50.0  # cm, the square's side
SIZE = 512  # px, the straightened view's side (10.24 px per cm); the encoders resize it themselves
MARKER_SIDE = 18.0  # cm, measured on the print
UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

FPS = 8  # frames per second kept in a clip
MAX_LENGTH = 6.0  # seconds
HAND_FREE = (0.1, 1.1)  # seconds after the clap: the hand is out of view (it came in 1.2-1.3 s after on the test take)
MARKER_DICTIONARY, MARKER_ID = cv2.aruco.DICT_4X4_50, 0
MAX_DRIFT = 1.0  # image px: the marker moving further than this means the camera moved
CLAP_GAP = 3.0  # seconds: claps closer than this are one clap (echo, double hit)
CLAP_RANGE = 10.0  # dB: a clap is at most this much quieter than the loudest one in the take
ENVELOPE = 0.01  # seconds per loudness sample


@dataclass
class Clip:
    name: str  # "<take>-<nn>", nn counting from 01, so names sort in take order
    clap: float  # seconds into the take
    timestamps: np.ndarray  # (T,) seconds since the clap, k / FPS
    source_times: np.ndarray  # (T,) seconds into the take of the video frame used for each
    frames: np.ndarray  # (T, size, size, 3) uint8, the straightened table square
    hand_free_frames: int  # video frames in the hand-free second after the clap
    hand_free_found: int  # how many of them show the marker
    corners: np.ndarray  # (4, 2) marker corners (TL, TR, BR, BL as printed), image px, mean over the hand-free second
    drift_px: np.ndarray  # (T,) per frame, how far the marker is from `corners` (largest corner distance; NaN = not found)
    step_px: float  # how far `corners` moved from the previous clip's (NaN for a take's first clip)
    image_to_table: np.ndarray  # (3, 3) homography: video image px -> table cm (from `corners`)

    def to_cm(self, points) -> np.ndarray:
        """Video image px (..., 2) -> table cm (..., 2), on the table plane."""
        points = np.asarray(points, dtype=np.float64)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.image_to_table).reshape(points.shape)

    @property
    def px_per_cm(self) -> float:
        """The video's scale at the marker: image px per table cm."""
        return float(np.linalg.norm(self.corners - np.roll(self.corners, 1, axis=0), axis=1).mean() / MARKER_SIDE)

    @property
    def drifted(self) -> bool:
        """The camera moved during the clip: its frames no longer match its homography."""
        return bool(np.nanmax(self.drift_px) > MAX_DRIFT)

    @property
    def camera_moved(self) -> bool:
        """The camera moved since the previous clip. Not a fault (each clip has its own homography), only a flag."""
        return bool(self.step_px > MAX_DRIFT)


@dataclass
class Take:
    name: str
    claps: np.ndarray  # (n,) seconds into the take
    clips: list[Clip]
    clap_db: np.ndarray  # (n,) each clap's loudness above the take's noise floor (median), dB
    other: tuple[float, float]  # the loudest sound more than CLAP_GAP from every clap: (seconds, dB above floor)
    frame_times: np.ndarray  # (N,) every video frame's time, seconds into the take


def cut_take(video: str | Path, square=setups.CARPET.square, size: int = SIZE) -> Take:
    """Cut a take into clips and straighten their frames to `size` x `size` px views of the table square
    (`square`: its corners in the marker's frame, see setups.Setup)."""
    video = Path(video)
    times, level = _loudness(video)
    claps = _claps(times, level)
    frame_times = _frame_times(video)

    # Each clip's frames: the video frame nearest each k / FPS after the clap, before the next clap.
    # Its hand-free second: every video frame in HAND_FREE after the clap.
    ends = np.append(claps[1:], np.inf)
    picks, hand_free = [], []
    for clap, end in zip(claps, ends):
        wanted = clap + np.arange(round(MAX_LENGTH * FPS)) / FPS
        wanted = wanted[(wanted < end) & (wanted <= frame_times[-1])]
        picks.append(_nearest(frame_times, wanted))
        hand_free.append(np.flatnonzero((frame_times >= clap + HAND_FREE[0]) & (frame_times < clap + HAND_FREE[1])))

    # First pass: the marker in every frame used. Each clip's homography comes from its hand-free second.
    corners = _marker_corners(video, frame_times, set(np.concatenate(hand_free + picks).tolist()))
    found = [[corners[i] for i in still if corners[i] is not None] for still in hand_free]
    for clap, f in zip(claps, found):
        if not f:
            raise ValueError(f"{video.name}: no marker in the hand-free second after the clap at {clap:.2f} s")
    references = [np.mean(f, axis=0) for f in found]
    image_to_table = [_image_to_table(reference, square) for reference in references]

    # Second pass: straighten each clip frame with its clip's homography.
    owners = defaultdict(list)  # frame -> clips that use it (at most two, where a clip ends at the next clap)
    for k, pick in enumerate(picks):
        for i in pick.tolist():
            owners[i].append(k)
    straight = [{} for _ in picks]
    for i, image in _frames(video, frame_times, set(owners), "rgb24"):
        for k in owners[i]:
            straight[k][i] = _straighten(image, image_to_table[k], size)

    clips = []
    for k, (clap, pick, still) in enumerate(zip(claps, picks, hand_free)):
        clips.append(
            Clip(
                name=f"{video.stem}-{k + 1:02d}",
                clap=float(clap),
                timestamps=np.arange(len(pick)) / FPS,
                source_times=frame_times[pick],
                frames=np.stack([straight[k][i] for i in pick.tolist()]),
                hand_free_frames=len(still),
                hand_free_found=len(found[k]),
                corners=references[k],
                drift_px=np.array([_shift(corners[i], references[k]) for i in pick.tolist()]),
                step_px=_shift(references[k - 1], references[k]) if k else np.nan,
                image_to_table=image_to_table[k],
            )
        )

    floor = np.median(level)
    clap_db = level[np.searchsorted(times, claps)] - floor
    away = np.all(np.abs(times[:, None] - claps[None, :]) > CLAP_GAP, axis=1)
    loudest = np.flatnonzero(away)[np.argmax(level[away])] if away.any() else None
    other = (float(times[loudest]), float(level[loudest] - floor)) if loudest is not None else (np.nan, np.nan)
    return Take(
        name=video.stem, claps=claps, clips=clips, clap_db=clap_db, other=other, frame_times=frame_times
    )


def _image_to_table(corners: np.ndarray, square) -> np.ndarray:
    """The homography from image px to table cm, given the marker's corners in the image and the square's in
    the marker's frame."""
    marker = (MARKER_SIDE * UNIT_SQUARE).astype(np.float32)
    marker_to_image = cv2.getPerspectiveTransform(marker, corners.astype(np.float32))
    square = np.asarray(square, dtype=np.float64).reshape(-1, 1, 2)
    square = cv2.perspectiveTransform(square, marker_to_image).astype(np.float32)
    return cv2.getPerspectiveTransform(square, (WORKSPACE * UNIT_SQUARE).astype(np.float32))


def _straighten(image: np.ndarray, image_to_table: np.ndarray, size: int) -> np.ndarray:
    """The table square as a `size` x `size` view; pixel (u, v) covers table cm [u, u + 1) * WORKSPACE / size.
    Warped at twice the size, then averaged down, so the 1.7x shrink from the 1080p frame does not alias."""
    scale = 2 * size / WORKSPACE
    table_to_view = np.array([[scale, 0, -0.5], [0, scale, -0.5], [0, 0, 1]])
    big = cv2.warpPerspective(image, table_to_view @ image_to_table, (2 * size, 2 * size), flags=cv2.INTER_LINEAR)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)


def _shift(corners: np.ndarray | None, reference: np.ndarray) -> float:
    """How far the marker moved from `reference`: the largest corner distance in px (NaN if not found)."""
    return np.nan if corners is None else float(np.linalg.norm(corners - reference, axis=1).max())


def _loudness(video: Path) -> tuple[np.ndarray, np.ndarray]:
    """The time at the middle of each 10 ms of the audio track, and its loudness there in dB (RMS)."""
    with av.open(str(video)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        mono = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks, start = [], None
        for frame in container.decode(stream):
            start = frame.time if start is None else start
            chunks += [f.to_ndarray().ravel() for f in mono.resample(frame)]
        chunks += [f.to_ndarray().ravel() for f in mono.resample(None)]
    sound = np.concatenate(chunks).astype(np.float64)
    width = round(ENVELOPE * rate)
    n = len(sound) // width
    rms = np.sqrt((sound[: n * width].reshape(n, width) ** 2).mean(axis=1))
    return start + (np.arange(n) + 0.5) * width / rate, 20 * np.log10(rms + 1e-12)


def _claps(times: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Loudest first: keep a peak if no louder peak is within CLAP_GAP, and if it is within CLAP_RANGE dB of
    the loudest. The second rule drops the click of the phone's stop button (17 dB below the claps on the
    test take)."""
    claps = []
    for i in np.argsort(level)[::-1]:
        if level[i] < level.max() - CLAP_RANGE:
            break
        if all(abs(times[i] - t) >= CLAP_GAP for t in claps):
            claps.append(times[i])
    return np.sort(np.array(claps))


def _frame_times(video: Path) -> np.ndarray:
    """Presentation time of every video frame, in order, read from the container without decoding."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        pts = [p.pts for p in container.demux(stream) if p.pts is not None]
        return np.sort(np.array(pts)) * float(stream.time_base)


def _nearest(times: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Index of the time nearest each wanted time (`times` sorted)."""
    after = np.clip(np.searchsorted(times, wanted), 1, len(times) - 1)
    before = after - 1
    return np.where(wanted - times[before] <= times[after] - wanted, before, after)


def _frames(video: Path, frame_times: np.ndarray, indices: set[int], pixels: str):
    """Yield (position, image) for the video frames at these positions (in presentation order).
    `pixels` is a PyAV format: "rgb24" or "gray"."""
    last = max(indices, default=-1)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # decode on several threads
        for i, frame in enumerate(container.decode(stream)):
            if i in indices:
                assert abs(frame.time - frame_times[i]) < 1e-3, "the decoder skipped a frame"
                yield i, frame.to_ndarray(format=pixels)
            if i >= last:
                break


def _marker_corners(video: Path, frame_times: np.ndarray, indices: set[int]) -> dict[int, np.ndarray | None]:
    """The marker's four corners (TL, TR, BR, BL as printed; (4, 2) image px) in each of these frames, or None
    where it is not found.

    AprilTag corner refinement: OpenCV's default settings missed the marker in 5 of the test take's 904 frames
    although it was plainly visible, this one in none. On a whole 1080p frame it takes 0.4 s, so the search
    runs on a crop around where the marker was last found (15 ms), and on the whole frame only when that fails.
    """
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    parameters.aprilTagQuadDecimate = 2.0  # look for the outline at half size: 4x faster, corner jitter 0.01 -> 0.02 px
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY), parameters)

    def find(gray, x0=0, y0=0):
        found, ids, _ = detector.detectMarkers(gray)
        ids = [] if ids is None else ids.ravel().tolist()
        return found[ids.index(MARKER_ID)][0].astype(np.float64) + [x0, y0] if MARKER_ID in ids else None

    corners, crop = {}, None
    for i, gray in _frames(video, frame_times, indices, "gray"):
        found = None
        if crop is not None:
            x0, y0, x1, y1 = crop
            found = find(gray[y0:y1, x0:x1], x0, y0)
        if found is None:
            found = find(gray)
        if found is not None:  # the crop: the marker plus half a marker side (9 cm) all round
            margin = np.ptp(found, axis=0).max() / 2
            (x0, y0), (x1, y1) = np.floor(found.min(axis=0) - margin), np.ceil(found.max(axis=0) + margin)
            crop = (max(int(x0), 0), max(int(y0), 0), int(x1), int(y1))
        corners[i] = found
    return corners
