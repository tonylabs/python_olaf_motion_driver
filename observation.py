"""Observation assembly — must match WalkingObservationsCfg.PolicyCfg exactly.

Concatenation order (see ``olaf_walking_env_cfg.py``):

    velocity_commands       (4)     — [lin_vel_x, lin_vel_y, ang_vel_z, heading]
    root_pose_in_path       (3)     — [dx_p, dy_p, dyaw]
    base_lin_vel            (3)     — root-frame linear velocity
    base_ang_vel            (3)     — root-frame angular velocity
    projected_gravity       (3)     — root-frame gravity unit vector
    joint_pos_rel           (12)    — q − q_default
    joint_vel               (12)
    last_action             (12)    — a_{t-1}
    prev_prev_action        (12)    — a_{t-2}
    gait_phase              (2)     — [sin(2πφ), cos(2πφ)]

Total: 66 floats.

``base_lin_vel`` on a Pi is trickier than in sim.  Options (from cheapest to
best): (1) leave zero — walking env was trained with a noise corrupt on this
channel so it degrades gracefully; (2) fuse IMU accelerometer with leg
kinematics via a Kalman filter; (3) external motion capture.  Start with
(1) and plan to upgrade to (2).
"""
from __future__ import annotations

import math

import numpy as np

from .config import (
    DEFAULT_JOINT_POS,
    GAIT_PERIOD_S,
    JOINT_ORDER,
    N_JOINTS,
    POLICY_DT,
)

OBS_DIM = 4 + 3 + 3 + 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS + N_JOINTS + 2


class PathFrame:
    """2-D path-frame integrator matching ``walking_imitation.py``.

    Advances (x_p, y_p, ψ_p) by integrating the commanded linear velocity
    (rotated into world via ψ_p) and commanded yaw rate.  Constrained to
    stay within ``max_offset_m`` of the torso to match training.
    """

    def __init__(self, max_offset_m: float = 0.5) -> None:
        self._state = np.zeros(3, dtype=np.float32)  # (x, y, yaw)
        self._max = max_offset_m

    def update(self, vx_cmd: float, vy_cmd: float, wz_cmd: float,
               torso_xy_world: np.ndarray, dt: float) -> None:
        yaw = self._state[2]
        c, s = math.cos(yaw), math.sin(yaw)
        dx =  c * vx_cmd - s * vy_cmd
        dy =  s * vx_cmd + c * vy_cmd
        self._state[0] += dx * dt
        self._state[1] += dy * dt
        self._state[2] += wz_cmd * dt
        # clamp within max_offset of torso
        off = self._state[:2] - torso_xy_world
        dist = float(np.linalg.norm(off))
        if dist > self._max:
            self._state[:2] = torso_xy_world + off * (self._max / dist)

    def root_pose_in_path(self, torso_xy_world: np.ndarray, yaw_world: float
                          ) -> np.ndarray:
        dx_w = torso_xy_world[0] - self._state[0]
        dy_w = torso_xy_world[1] - self._state[1]
        yaw_p = self._state[2]
        c, s = math.cos(yaw_p), math.sin(yaw_p)
        dx_p =  c * dx_w + s * dy_w
        dy_p = -s * dx_w + c * dy_w
        dyaw = math.atan2(math.sin(yaw_world - yaw_p),
                          math.cos(yaw_world - yaw_p))
        return np.array([dx_p, dy_p, dyaw], dtype=np.float32)


class ObservationBuilder:
    def __init__(self) -> None:
        self.last_action     = np.zeros(N_JOINTS, dtype=np.float32)
        self.prev_prev_action = np.zeros(N_JOINTS, dtype=np.float32)
        self._phi = 0.0
        self._path = PathFrame()
        assert len(JOINT_ORDER) == N_JOINTS

    def step_phase(self) -> None:
        self._phi = (self._phi + POLICY_DT / GAIT_PERIOD_S) % 1.0

    def build(self,
              velocity_cmd: np.ndarray,        # (4,) [vx, vy, wz, heading]
              torso_xy_world: np.ndarray,      # (2,)
              yaw_world: float,
              base_lin_vel_root: np.ndarray,   # (3,)
              base_ang_vel_root: np.ndarray,   # (3,)
              projected_gravity: np.ndarray,   # (3,)
              joint_pos: np.ndarray,           # (12,) URDF frame
              joint_vel: np.ndarray,           # (12,)
              ) -> np.ndarray:
        # Advance path frame with commanded velocity.
        self._path.update(
            vx_cmd=float(velocity_cmd[0]),
            vy_cmd=float(velocity_cmd[1]),
            wz_cmd=float(velocity_cmd[2]),
            torso_xy_world=torso_xy_world,
            dt=POLICY_DT,
        )
        rpp = self._path.root_pose_in_path(torso_xy_world, yaw_world)

        two_pi_phi = 2.0 * math.pi * self._phi
        gait = np.array([math.sin(two_pi_phi), math.cos(two_pi_phi)],
                        dtype=np.float32)

        joint_pos_rel = (joint_pos - DEFAULT_JOINT_POS).astype(np.float32)

        obs = np.concatenate([
            velocity_cmd.astype(np.float32),
            rpp,
            base_lin_vel_root.astype(np.float32),
            base_ang_vel_root.astype(np.float32),
            projected_gravity.astype(np.float32),
            joint_pos_rel,
            joint_vel.astype(np.float32),
            self.last_action,
            self.prev_prev_action,
            gait,
        ])
        assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape} != {OBS_DIM}"
        return obs

    def push_action(self, action: np.ndarray) -> None:
        self.prev_prev_action = self.last_action.copy()
        self.last_action = action.astype(np.float32).copy()
