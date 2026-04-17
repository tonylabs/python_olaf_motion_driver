"""Observation assembly — must match the deployed ONNX policy input.

Concatenation order (48 floats, verified against policy.onnx input shape):

    velocity_commands       (4)     — [lin_vel_x, lin_vel_y, ang_vel_z, heading]
    base_ang_vel            (3)     — root-frame angular velocity
    projected_gravity       (3)     — root-frame gravity unit vector
    joint_pos_rel           (12)    — q − q_default
    joint_vel               (12)
    last_action             (12)    — a_{t-1}
    gait_phase              (2)     — [sin(2πφ), cos(2πφ)]

Total: 48 floats.
"""
from __future__ import annotations

import math

import numpy as np

from config import (
    DEFAULT_JOINT_POS,
    GAIT_PERIOD_S,
    JOINT_ORDER,
    N_JOINTS,
    POLICY_DT,
)

OBS_DIM = 4 + 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS + 2  # 48


class ObservationBuilder:
    def __init__(self) -> None:
        self.last_action = np.zeros(N_JOINTS, dtype=np.float32)
        self._phi = 0.0
        assert len(JOINT_ORDER) == N_JOINTS

    def step_phase(self) -> None:
        self._phi = (self._phi + POLICY_DT / GAIT_PERIOD_S) % 1.0

    def build(self,
              velocity_cmd: np.ndarray,        # (4,) [vx, vy, wz, heading]
              base_ang_vel_root: np.ndarray,   # (3,)
              projected_gravity: np.ndarray,   # (3,)
              joint_pos: np.ndarray,           # (12,) URDF frame
              joint_vel: np.ndarray,           # (12,)
              ) -> np.ndarray:
        two_pi_phi = 2.0 * math.pi * self._phi
        gait = np.array([math.sin(two_pi_phi), math.cos(two_pi_phi)],
                        dtype=np.float32)

        joint_pos_rel = (joint_pos - DEFAULT_JOINT_POS).astype(np.float32)

        obs = np.concatenate([
            velocity_cmd.astype(np.float32),
            base_ang_vel_root.astype(np.float32),
            projected_gravity.astype(np.float32),
            joint_pos_rel,
            joint_vel.astype(np.float32),
            self.last_action,
            gait,
        ])
        assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape} != {OBS_DIM}"
        return obs

    def push_action(self, action: np.ndarray) -> None:
        self.last_action = action.astype(np.float32).copy()
