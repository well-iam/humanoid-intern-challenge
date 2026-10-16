"""Copy the figures and GIFs the README shows from the run folder into assets/.

Rerun after any new run to refresh them. Each file must stay under 5 MB so the README loads quickly on GitHub;
a bigger one stops the script, and is shrunk with --size on the script that drew it.
"""

import argparse
import shutil
from pathlib import Path

from pwm import stages

LIMIT = 5e6  # bytes

# README asset -> file in the run folder
ASSETS = {
    "gif-strip.gif": "clips/gif-strip/VID_20261016_141304-11.gif",
    "pusht-error-vs-horizon.png": "pusht/error-vs-horizon.png",
    "pusht-seeds.png": "pusht/dinov2/error-vs-horizon-seeds.png",
    "ranking.png": "pusht/dinov2/ranking/ranking.png",
    "pusht-panda.gif": "pusht/panda/pusht-002.gif",
    "clips-error-vs-horizon.png": "clips/dinov2/error-vs-horizon.png",
    "near-miss.png": "clips/dinov2/near-miss.png",
    "probe-scaling.png": "clips/dinov2/probe-scaling.png",
    "panda-push.gif": "clips/panda/VID_20261016_163841-02.gif",
    "panda-near-miss.gif": "clips/panda/VID_20261016_165020-05.gif",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

root = stages.data_root(args.root)
out = Path(__file__).resolve().parents[1] / "assets"
out.mkdir(exist_ok=True)
for name, source in ASSETS.items():
    size = (root / source).stat().st_size
    if size > LIMIT:
        raise SystemExit(f"{source} is {size / 1e6:.1f} MB, over {LIMIT / 1e6:.0f} MB: draw it smaller first")
    shutil.copy2(root / source, out / name)
    print(f"{size / 1e6:5.2f} MB  assets/{name}  <-  {source}")
