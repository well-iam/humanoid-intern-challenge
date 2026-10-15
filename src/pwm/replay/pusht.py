"""The Push-T judge: Push-T's own pymunk simulator (gym-pusht), the physics that made the data.

The pusher path is resampled to the simulator's 10 Hz control rate. Push-T moves the pusher
with a PD controller towards an absolute target, so a pusher sent straight to each path
position lags about 13 units behind it. Instead, each target is chosen so that the pusher
lands exactly on the next path position: the controller is linear in (position, velocity,
target), so one division inverts it. Replaying recorded pusher paths then reproduces the
recorded final T position to a median of 0.0 and a worst case of 1.4 units, against a
median of 4.4 and a worst case of 258 units with the path positions as targets.

The start pose is set directly on the bodies; gym-pusht's `reset_to_state` option is not
used because it sets the angle after the position, which moves the T.
"""

import numpy as np
from gym_pusht.envs.pusht import PushTEnv

from pwm.datasets.pusht import T_CENTRE_OFFSET, UNIT, t_centre
from pwm.replay import Push, Replayed

CONTROL_HZ = 10


class PushTJudge:
    def replay(self, push: Push, frame_size: int | None = None) -> Replayed:
        assert push.shape == "T", f"the Push-T judge only has the T, not {push.shape!r}"
        assert push.unit == UNIT, f"the Push-T judge works in {UNIT}, not {push.unit!r}"

        env = PushTEnv(obs_type="state" if frame_size is None else "pixels",
                       observation_width=frame_size or 96, observation_height=frame_size or 96)
        env.reset(seed=0)
        x, y, angle = push.start
        env.block.angle = angle
        env.block.position = (x + T_CENTRE_OFFSET * np.sin(angle), y - T_CENTRE_OFFSET * np.cos(angle))
        env.agent.position = tuple(push.path[0])
        # Start at rest: reset() puts the bodies at arbitrary places, where they may have collided.
        env.agent.velocity = env.block.velocity = (0, 0)
        env.block.angular_velocity = 0
        # Move the shapes with their bodies now, not at the first physics step: otherwise the first
        # frame shows the old pose and stale contacts nudge the T (worst replay error 4.3 -> 1.4).
        for body in (env.agent, env.block):
            env.space.reindex_shapes_for_body(body)

        t0, t1 = push.timestamps[0], push.timestamps[-1]
        ticks = t0 + np.arange(int(np.ceil((t1 - t0) * CONTROL_HZ - 1e-9)) + 1) / CONTROL_HZ
        waypoints = np.stack([np.interp(ticks, push.timestamps, push.path[:, i]) for i in (0, 1)], axis=1)

        states, frames = [_state(env)], [env.get_obs()] if frame_size else None
        for waypoint in waypoints[1:]:
            obs = env.step(_controller_target(env, waypoint))[0]
            states.append(_state(env))
            if frame_size:
                frames.append(obs)
        env.close()

        states = np.array(states)
        poses = np.concatenate([t_centre(states[:, 2:5]), states[:, 4:5]], axis=1)
        at = lambda a: np.stack([np.interp(push.timestamps, ticks, a[:, i]) for i in range(a.shape[1])], axis=1)
        poses = at(poses)
        nearest = np.abs(push.timestamps[:, None] - ticks[None]).argmin(axis=1)
        return Replayed(
            final=poses[-1],
            poses=poses,
            pusher=at(states[:, :2]),
            frames=np.array(frames)[nearest] if frame_size else None,
        )


def _controller_target(env: PushTEnv, position: np.ndarray) -> np.ndarray:
    """The PD target that brings the pusher from its current position and velocity to `position`
    after one control step."""
    x, v = np.array(env.agent.position), np.array(env.agent.velocity)
    from_x, from_v, from_target = (_one_control_step(env, *unit) for unit in np.eye(3))
    return (position - from_x * x - from_v * v) / from_target


def _one_control_step(env: PushTEnv, x: float, v: float, target: float) -> float:
    """Pusher position after one control step of Push-T's PD controller (one axis, no contact:
    the pusher is kinematic, so the T never pushes back)."""
    for _ in range(round(1 / (env.dt * env.control_hz))):
        v += (env.k_p * (target - x) - env.k_v * v) * env.dt
        x += v * env.dt
    return x


def _state(env: PushTEnv) -> np.ndarray:
    """Agent x, y and block x, y, angle (the stored Push-T state, angle not wrapped)."""
    return np.array([*env.agent.position, *env.block.position, env.block.angle])
