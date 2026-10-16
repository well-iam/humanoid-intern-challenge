"""What the public Hugging Face dataset holds, picked from the run folder."""

from pwm import hub

KEPT = [
    "clips/episodes/VID_1-01.npz",  # my clips' tracks and straightened frames
    "clips/dinov2/embeddings/VID_1-01.npy",
    "clips/dinov2/predictor.pt",
    "clips/dinov2/predictor-blind.pt",
    "clips/dinov2/results.json",
    "clips/dinov2/near-miss.json",
    "clips/dinov2/probe-scaling.json",
    "clips/dinov2/only-takes/results.json",
    "clips/dinov2-30clips/results.json",  # the first 30-clip run, for the before/after
    "clips/dinov2-30clips/predictor.pt",
    "clips/panda-30clips/replay.json",
    "clips/panda/replay.json",
    "clips/panda/VID_1-01.gif",
    "clips/gif-strip/VID_1-01.gif",
    "clips/tracking/report.md",
    "clips/logs/train-predictor.log",
    "pusht/dinov2/embeddings/pusht-000.npy",
    "pusht/dinov2/seed-1/predictor.pt",
    "pusht/dinov2/ranking/ranking.json",
    "pusht/vjepa2/embeddings/pusht-000.npy",
    "pusht/vjepa2/results.json",
    "pusht/winner.json",
    "pusht/judge/noise-floor.json",
    "pusht/panda/pusht-002.gif",
    "pusht/logs/seeds/train-dinov2-seed1.log",
]
LEFT_OUT = [
    "clips/raw/VID_1.mp4",  # the full-size take: the dataset gets a 720p copy instead
    "clips/test-take/VID_0.mp4",  # test takes are setup checks, never project data
    "clips/check/VID_0/report.md",
    "clips-test-take/episodes/VID_0-01.npz",
    "clips/toy/results.json",
    "clips/logs/retrievals.log",  # logs other than training curves hold local paths
    "clips/logs/per_clip.py",
    "pusht/raw/pusht.zip",  # Push-T is fetched from its original URL, never re-hosted
    "pusht/episodes/pusht-000.npz",
    "pusht/dinov2/embeddings/pusht-001.partial.npy",  # an interrupted write
    "models/mediapipe/hand_landmarker.task",
    "clips/dinov2-30clips/embeddings/VID_1-01.npy",  # the same as clips/dinov2/embeddings
    "clips/episodes-30clips/VID_1-01.npz",  # the same as clips/episodes
    "clips/tracking-30clips/report.md",
]


def test_the_public_dataset_holds_my_clips_and_both_runs_but_no_test_takes_and_no_push_t_source(tmp_path):
    for path in KEPT + LEFT_OUT:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(path)

    files = hub.public_files(tmp_path)

    assert sorted(files) == sorted(KEPT)
    assert all(files[path] == tmp_path / path for path in KEPT)
    assert hub.public_videos(tmp_path) == {"clips/videos/VID_1.mp4": tmp_path / "clips/raw/VID_1.mp4"}
