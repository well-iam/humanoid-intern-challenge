"""The Panda replay: a simulated Franka Panda in MuJoCo pushes the object along a pusher path.

The Menagerie Panda stands on the table at the origin, facing +x. Its closed fingers point
straight down with their tips `TIP_HEIGHT` above the table, and follow the pusher path:
mink inverse kinematics turns each path point into joint targets for the Panda's own joint
position servos. Those servos lag behind a moving target by their damping over stiffness
(0.1 s, so 1 cm at 10 cm/s), so the joint velocity from the inverse kinematics is added as a
feed-forward term. The arm links also get gravity compensation, as on the real robot.

`floating=True` switches to the fallback: the same hand with no arm, moved exactly along the
path as a mocap body (a body the simulator moves where it is told, however hard it pushes).

Table to world: the workspace square is about 50 cm wide (Push-T's 512 units become 50 cm;
centimetres stay centimetres), centred `REACH` in front of the robot. The table's x axis
(image right) is the world's -y and its y axis (image down) is the world's -x, so the robot
stands below the image, reaching up into it. Seen from above, the table frame with y down is
a mirror image of the world frame, so a table angle a is the world yaw -pi/2 - a.

`slowdown` runs the replay that many times slower than the recording, for paths too fast
for the arm; poses are still reported at the recorded timestamps.

Rendering is headless: on Linux without a display, MUJOCO_GL defaults to egl (Colab); set
MUJOCO_GL=osmesa where EGL is missing. Windows and macOS need nothing.
"""

import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mink  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pwm.replay import Push, Replayed  # noqa: E402
from pwm.replay.menagerie import panda_xml  # noqa: E402

REACH = 0.5  # m, from the robot base to the centre of the workspace
WORKSPACE_METRES = 0.5  # width of the Push-T workspace on the simulated table
METRES_PER_UNIT = {"cm": 0.01}  # Push-T units: WORKSPACE_METRES / workspace
TIP_HEIGHT = 0.015  # m, fingertips above the table
FINGERTIP_DEPTH = 0.113  # m, from the hand frame's origin down to the closed fingertips
HAND_DOWN = np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2)  # hand pointing down, fingers along world y
T_HEIGHT = 0.03  # m, height of the simulated Push-T T
DENSITY = 500.0  # kg/m3, a light block: 6 x 8 x 3 cm weighs 72 g
CONTROL_HZ = 100
CAMERA_HEIGHT = 1.5  # m above the table


class PandaReplay:
    def __init__(self, floating: bool = False, slowdown: float = 1.0, camera: str = "top", root=None):
        from pwm.stages import data_root

        assert camera in ("top", "side"), camera
        self.floating = floating
        self.slowdown = slowdown
        self.camera = camera
        self.xml = panda_xml(data_root(root))

    def replay(self, push: Push, frame_size: int | None = None) -> Replayed:
        table = _Table(push)
        model = self._scene(push, table, frame_size)
        data = mujoco.MjData(model)
        tip = model.site("pusher").id
        obj = model.body("object")

        t0 = push.timestamps[0]
        arm = None if self.floating else _ArmControl(self.xml, model)
        start = table.to_world(push.path[0])
        if arm:
            arm.reset(data, start)
        else:
            data.mocap_pos[0] = _hand_position(start)
        free = model.jnt_qposadr[obj.jntadr[0]]
        data.qpos[free : free + 7] = [*table.object_world(push.start), *table.object_quat(push.start[2])]
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, frame_size, frame_size) if frame_size else None
        record_at = (push.timestamps - t0) * self.slowdown
        steps_per_control = max(1, round(1 / (CONTROL_HZ * model.opt.timestep)))
        poses, pusher, frames = [], [], []
        step = 0
        for when in record_at:
            while data.time < when - model.opt.timestep / 2:
                if step % steps_per_control == 0:
                    t = t0 + (data.time + steps_per_control * model.opt.timestep) / self.slowdown
                    target = table.to_world(_interp(push, t))
                    if arm:
                        arm.control(data, target, steps_per_control * model.opt.timestep)
                    else:
                        data.mocap_pos[0] = _hand_position(target)
                mujoco.mj_step(model, data)
                step += 1
            pusher.append(table.from_world(data.site_xpos[tip][:2]))
            poses.append(table.object_pose(data.xpos[obj.id][:2], data.xquat[obj.id]))
            if renderer:
                renderer.update_scene(data, camera=self.camera)
                if self.camera == "top":
                    _ghost_arm(renderer.scene, model)
                frames.append(renderer.render().copy())
        if renderer:
            renderer.close()

        poses = np.array(poses)
        poses[:, 2] = push.start[2] + np.unwrap(poses[:, 2] - push.start[2])
        return Replayed(final=poses[-1], poses=poses, pusher=np.array(pusher),
                        frames=np.array(frames) if frame_size else None)

    def _scene(self, push: Push, table: "_Table", frame_size: int | None) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(self.xml))
        hand = spec.body("hand")
        hand.add_site(name="pusher", pos=[0, 0, FINGERTIP_DEPTH], size=[0.005, 0, 0], rgba=[1, 0, 0, 1])
        if self.floating:
            spec = _floating_hand(spec)
        else:
            for body in spec.bodies:
                body.gravcomp = 1

        world = spec.worldbody
        for light in list(spec.lights):
            spec.delete(light)
        spec.visual.headlight.ambient = [0.25] * 3
        spec.visual.headlight.diffuse = [0.15] * 3
        world.add_light(pos=[REACH, 0, 2], dir=[0, 0, -1], diffuse=[0.45] * 3, specular=[0.05] * 3)
        world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.01], rgba=[1.0, 1.0, 0.97, 1])
        fovy = 2 * np.degrees(np.arctan(table.half_width / CAMERA_HEIGHT))
        world.add_camera(name="top", pos=[REACH, 0, CAMERA_HEIGHT], xyaxes=[0, -1, 0, 1, 0, 0], fovy=fovy)
        world.add_camera(name="side", pos=[REACH + 0.75, -0.75, 0.7], xyaxes=_look_at([REACH + 0.75, -0.75, 0.7], [REACH - 0.1, 0, 0.1]), fovy=45)

        body = world.add_body(name="object")
        body.add_freejoint()
        for centre, half in table.object_boxes(push):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre, size=half, density=DENSITY,
                          rgba=[0.47, 0.53, 0.6, 1])
        if frame_size:
            spec.visual.global_.offwidth = max(640, frame_size)
            spec.visual.global_.offheight = max(480, frame_size)
        return spec.compile()


class _ArmControl:
    """Inverse kinematics on the arm alone (mink), sent to the arm's joint servos with a
    velocity feed-forward term."""

    def __init__(self, xml, model: mujoco.MjModel):
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.body("hand").add_site(name="pusher", pos=[0, 0, FINGERTIP_DEPTH])
        self.ik_model = spec.compile()
        self.config = mink.Configuration(self.ik_model)
        self.task = mink.FrameTask("pusher", "site", position_cost=1.0, orientation_cost=1.0)
        self.posture = mink.PostureTask(self.ik_model, cost=1e-2)
        self.limits = [mink.ConfigurationLimit(self.ik_model)]
        self.config.update_from_keyframe("home")
        self.posture.set_target_from_configuration(self.config)
        self.arm = slice(0, 7)
        # Servo force = kp * (ctrl - q) - kv * qdot, so ctrl = q + (kv / kp) * qdot tracks a moving q.
        self.lead = -model.actuator_biasprm[:7, 2] / model.actuator_gainprm[:7, 0]
        self.ctrlrange = model.actuator_ctrlrange[:7]

    def _solve(self, target: np.ndarray, dt: float, iterations: int) -> np.ndarray:
        self.task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(HAND_DOWN), np.array([*target, TIP_HEIGHT])))
        before = self.config.q[self.arm].copy()
        for _ in range(iterations):
            velocity = mink.solve_ik(self.config, [self.task, self.posture], dt, "daqp", damping=1e-3,
                                     limits=self.limits)
            self.config.integrate_inplace(velocity, dt)
        return (self.config.q[self.arm] - before) / (dt * iterations)

    def reset(self, data: mujoco.MjData, target: np.ndarray) -> None:
        self._solve(target, 0.01, 300)
        data.qpos[:9] = [*self.config.q[self.arm], 0, 0]
        data.ctrl[:7] = self.config.q[self.arm]
        data.ctrl[7] = 0  # fingers closed

    def control(self, data: mujoco.MjData, target: np.ndarray, dt: float) -> None:
        velocity = self._solve(target, dt, 1)
        q = self.config.q[self.arm]
        data.ctrl[:7] = np.clip(q + self.lead * velocity, *self.ctrlrange.T)
        data.ctrl[7] = 0


def _floating_hand(panda: mujoco.MjSpec) -> mujoco.MjSpec:
    """A scene with only the Panda's hand, fingers fixed closed, on a mocap body."""
    for name in ("finger_joint1", "finger_joint2"):
        panda.delete(panda.joint(name))
    for element in [*panda.tendons, *panda.equalities, *panda.actuators, *panda.keys]:
        panda.delete(element)
    spec = mujoco.MjSpec()
    spec.option.integrator = panda.option.integrator
    mocap = spec.worldbody.add_body(name="gripper", mocap=True, quat=HAND_DOWN)
    hand = panda.body("hand")
    hand.pos, hand.quat = [0, 0, 0], [1, 0, 0, 0]  # in the mocap body's frame, not on link 7
    mocap.add_frame().attach_body(hand, "", "")
    return spec


def _ghost_arm(scene: mujoco.MjvScene, model: mujoco.MjModel) -> None:
    """Draw the arm's links see-through in the top view, so they do not hide the object."""
    for geom in scene.geoms[: scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.body(model.geom_bodyid[geom.objid]).name.startswith("link"):
            geom.rgba[3] = 0.15


def _look_at(eye, target) -> list[float]:
    """Camera x and y axes for a camera at `eye` looking at `target`, with the world's z up."""
    forward = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    return [*right, *np.cross(right, forward)]


def _hand_position(target: np.ndarray) -> np.ndarray:
    return np.array([*target, TIP_HEIGHT + FINGERTIP_DEPTH])


def _interp(push: Push, t: float) -> np.ndarray:
    return np.array([np.interp(t, push.timestamps, push.path[:, i]) for i in (0, 1)])


class _Table:
    """Table units <-> world metres for one push."""

    def __init__(self, push: Push):
        self.scale = METRES_PER_UNIT.get(push.unit, WORKSPACE_METRES / push.workspace)
        self.centre = push.workspace / 2
        self.half_width = push.workspace / 2 * self.scale

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        u, v = uv
        return np.array([REACH - self.scale * (v - self.centre), -self.scale * (u - self.centre)])

    def from_world(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy
        return np.array([self.centre - y / self.scale, self.centre + (REACH - x) / self.scale])

    def object_boxes(self, push: Push) -> list[tuple[list[float], list[float]]]:
        """(centre, half sizes) of the object's boxes in its own frame, in metres. The body's
        x axis is the table's x and its y axis is the table's -y (z stays up)."""
        s = self.scale
        if push.shape == "T":
            h = T_HEIGHT / 2
            return [([0, 30 * s, h], [60 * s, 15 * s, h]), ([0, -30 * s, h], [15 * s, 45 * s, h])]
        assert push.shape == "box", f"unknown object shape {push.shape!r}"
        width, length, height = push.size
        return [([0, 0, height * s / 2], [width * s / 2, length * s / 2, height * s / 2])]

    def object_world(self, pose: np.ndarray) -> np.ndarray:
        return np.array([*self.to_world(pose[:2]), 0.0])

    def object_quat(self, angle: float) -> np.ndarray:
        yaw = -np.pi / 2 - angle
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    def object_pose(self, xy: np.ndarray, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([*self.from_world(xy), -np.pi / 2 - yaw])
