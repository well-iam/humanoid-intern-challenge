"""The MuJoCo Menagerie Panda, fetched once at a pinned commit into <root>/models/.

Only `panda.xml` and the mesh files it names are downloaded (about 35 MB), not the whole
Menagerie repository.
"""

import re
import urllib.request
from pathlib import Path

COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"  # 2026-07-04, the latest change to the Panda
URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/franka_emika_panda"
FILES = ["panda.xml", "LICENSE"]


def panda_xml(root: Path) -> Path:
    """Path of the Menagerie `panda.xml`, downloading the model on first use."""
    folder = Path(root) / "models" / f"menagerie-{COMMIT[:7]}" / "franka_emika_panda"
    xml = folder / "panda.xml"
    if not (folder / "complete").exists():
        (folder / "assets").mkdir(parents=True, exist_ok=True)
        for name in FILES:
            urllib.request.urlretrieve(f"{URL}/{name}", folder / name)
        for mesh in sorted(set(re.findall(r'file="([^"]+)"', xml.read_text()))):
            urllib.request.urlretrieve(f"{URL}/assets/{mesh}", folder / "assets" / mesh)
        (folder / "complete").touch()
    return xml
