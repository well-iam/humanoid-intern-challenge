"""Headless MuJoCo rendering check, for a machine with no display (Colab).

1. Renders a bare MuJoCo scene (a box on a plane): does the OpenGL backend work at all?
2. Replays a straight 20 cm push of the 6 x 8 x 3 cm box with the Panda and renders it from
   above and from the side: do the Menagerie download, mink and the replay's own cameras work?

Saves the last frames as <root>/render_check.png and exits non-zero on the first failure.
Pick the backend with MUJOCO_GL (egl or osmesa); on Linux without a display it defaults to egl.
"""

import argparse
import os
import sys
import time

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pwm import stages  # noqa: E402
from pwm.replay import Push, get_replay  # noqa: E402

BARE_SCENE = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/></body>
  </worldbody>
</mujoco>
"""

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", help="data root for the Menagerie model (default: $PWM_DATA, else ./data)")
args = parser.parse_args()
root = stages.data_root(args.root)
print(f"mujoco {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}")

model = mujoco.MjModel.from_xml_string(BARE_SCENE)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, 64, 64) as renderer:
    renderer.update_scene(data)
    bare = renderer.render()
assert bare.std() > 1, "the bare scene rendered as one flat colour"
print(f"1. bare scene: OK, {bare.shape}, pixel std {bare.std():.1f}")

timestamps = np.arange(0, 4.0 + 1e-9, 0.1)
path = np.stack([np.full_like(timestamps, 25.0), 42.0 - 5.0 * timestamps], axis=1)
push = Push(path=path, timestamps=timestamps, start=np.array([25.0, 25.0, 0.0]), shape="box", unit="cm",
            workspace=50.0, size=(6.0, 8.0, 3.0))
panels = []
for camera in ("top", "side"):
    started = time.perf_counter()
    result = get_replay("panda", camera=camera, root=root).replay(push, frame_size=128)
    moved = np.linalg.norm(result.final[:2] - push.start[:2])
    assert result.frames.std() > 1, f"the {camera} view rendered as one flat colour"
    assert moved > 3, f"the box moved only {moved:.1f} cm"
    print(f"2. Panda push, {camera} view: OK, box moved {moved:.1f} cm, "
          f"{len(result.frames)} frames in {time.perf_counter() - started:.1f} s")
    panels.append(result.frames[-1])

root.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.concatenate(panels, axis=1)).save(root / "render_check.png")
print(f"saved {root / 'render_check.png'}")
