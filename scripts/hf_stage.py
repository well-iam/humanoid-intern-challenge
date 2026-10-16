"""Build the Hugging Face dataset as a local folder, exactly as it will be uploaded (scripts/hf_upload.py).

Run-folder files are hard-linked (copied if the folder is on another drive), my takes are copied at 720p H.264,
and the tracks table, the split and the dataset card (docs/hf-dataset-card.md) are written. What goes in is
decided by pwm.hub. The folder is rebuilt from scratch on every run.

    uv run python scripts/hf_stage.py --out C:/Users/william.notaro/dev/pwm-data/hf-staging
"""

import argparse
import os
import shutil
from pathlib import Path

from pwm import hub, stages
from pwm.episodes import load_episodes

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", required=True, help="the staging folder (deleted and rebuilt)")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()
root, out = stages.data_root(args.root), Path(args.out)
assert out.resolve() != root.resolve() and out.resolve() not in root.resolve().parents, "--out would delete the data"
if out.exists():
    shutil.rmtree(out)

for name, source in hub.public_files(root).items():
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)

for name, take in hub.public_videos(root).items():
    print(f"720p copy of {take.name}")
    hub.transcode(take, out / name)

episodes = load_episodes(root / "clips" / "episodes")
hub.write_tracks(episodes, out / "clips" / "tracks.csv")
hub.write_split(episodes, out / "clips" / "split.json")
shutil.copy(Path(__file__).parents[1] / "docs" / "hf-dataset-card.md", out / "README.md")

files = sorted(p for p in out.rglob("*") if p.is_file())
print(f"{out}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")
