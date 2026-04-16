"""Wiring / kinematics constants for the physical OLAF.

Keep in sync with ``olaf_walking_env_cfg.py``.  The ORDER of ``JOINT_ORDER``
is the contract between policy outputs, observations, and motor commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Joint order — MUST match LEG_JOINT_NAMES in olaf_standing_env_cfg.py
# ---------------------------------------------------------------------------
JOINT_ORDER = (
    "l_hip_yaw_joint",
    "l_hip_roll_joint",
    "l_hip_pitch_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_yaw_joint",
    "r_hip_roll_joint",
    "r_hip_pitch_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
)
N_JOINTS = len(JOINT_ORDER)

# Default joint targets — action = raw_action + DEFAULT_JOINT_POS (scale=1.0,
# use_default_offset=True).  Must match init_state.joint_pos in env.
DEFAULT_JOINT_POS = np.array([
    0.00, -0.10,  0.90,  1.65,  0.70,  0.00,   # left leg
    0.00, -0.10, -0.90, -1.65,  0.70,  0.00,   # right leg
], dtype=np.float32)

# URDF joint limits (lower, upper) in the same order as JOINT_ORDER.
JOINT_LIMITS = np.array([
    (-0.10,  0.10),  # l_hip_yaw
    (-0.20,  0.20),  # l_hip_roll
    (-0.10,  1.20),  # l_hip_pitch
    ( 0.00,  1.80),  # l_knee_pitch
    (-0.65,  1.05),  # l_ankle_pitch
    (-0.26,  0.20),  # l_ankle_roll
    (-0.10,  0.10),  # r_hip_yaw
    (-0.20,  0.20),  # r_hip_roll
    (-1.20,  0.10),  # r_hip_pitch
    (-1.80,  0.00),  # r_knee_pitch
    (-0.65,  1.05),  # r_ankle_pitch
    (-0.20,  0.26),  # r_ankle_roll
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Motor hardware classes
# ---------------------------------------------------------------------------
class MotorKind(Enum):
    ROBSTRIDE_RS02 = "rs02"   # hip yaw
    ROBSTRIDE_RS03 = "rs03"   # hip roll/pitch, knee
    DAMIAO_J4340   = "dm4340" # ankle pitch
    DAMIAO_J4310   = "dm4310" # ankle roll


@dataclass(frozen=True)
class MotorSpec:
    can_id: int             # arbitration ID on the bus
    kind: MotorKind
    direction: int = 1      # +1 or -1 to flip sign vs. URDF convention
    zero_offset_rad: float = 0.0  # mechanical zero offset (encoder − URDF zero)
    kp: float = 0.0
    kd: float = 0.0


# ------------------ FILL IN WITH YOUR WIRING ------------------
# Example IDs — replace with your actual assignments.  Order does NOT matter
# in this dict; lookup is by joint name.
MOTOR_TABLE: dict[str, MotorSpec] = {
    "l_hip_yaw_joint":     MotorSpec(can_id=0x11, kind=MotorKind.ROBSTRIDE_RS02,
                                     kp=40.0, kd=1.0),
    "l_hip_roll_joint":    MotorSpec(can_id=0x12, kind=MotorKind.ROBSTRIDE_RS03,
                                     kp=50.0, kd=2.0),
    "l_hip_pitch_joint":   MotorSpec(can_id=0x13, kind=MotorKind.ROBSTRIDE_RS03,
                                     kp=80.0, kd=4.0),
    "l_knee_pitch_joint":  MotorSpec(can_id=0x14, kind=MotorKind.ROBSTRIDE_RS03,
                                     kp=80.0, kd=4.0),
    "l_ankle_pitch_joint": MotorSpec(can_id=0x15, kind=MotorKind.DAMIAO_J4340,
                                     kp=30.0, kd=2.0),
    "l_ankle_roll_joint":  MotorSpec(can_id=0x16, kind=MotorKind.DAMIAO_J4310,
                                     kp=8.0,  kd=2.0),
    "r_hip_yaw_joint":     MotorSpec(can_id=0x21, kind=MotorKind.ROBSTRIDE_RS02,
                                     kp=40.0, kd=1.0),
    "r_hip_roll_joint":    MotorSpec(can_id=0x22, kind=MotorKind.ROBSTRIDE_RS03,
                                     kp=50.0, kd=2.0),
    "r_hip_pitch_joint":   MotorSpec(can_id=0x23, kind=MotorKind.ROBSTRIDE_RS03,
                                     kp=80.0, kd=4.0),
    "r_knee_pitch_joint":  MotorSpec(can_id=0x24, kind=MotorKind.ROBSTRIDE_RS03,
                                     kp=80.0, kd=4.0),
    "r_ankle_pitch_joint": MotorSpec(can_id=0x25, kind=MotorKind.DAMIAO_J4340,
                                     kp=30.0, kd=2.0),
    "r_ankle_roll_joint":  MotorSpec(can_id=0x26, kind=MotorKind.DAMIAO_J4310,
                                     kp=8.0,  kd=2.0),
}


# ---------------------------------------------------------------------------
# Loop rates
# ---------------------------------------------------------------------------
POLICY_HZ       = 50.0
MOTOR_HZ        = 600.0
POLICY_DT       = 1.0 / POLICY_HZ
MOTOR_DT        = 1.0 / MOTOR_HZ
LPF_CUTOFF_HZ   = 37.5                      # per paper §VII-C
GAIT_PERIOD_S   = 0.6                       # must match ObsTerm gait_phase params

# Safety
POLICY_WATCHDOG_S = 0.040                   # drop to damping if policy misses
KP_RAMP_S         = 2.0                     # soft-start ramp on startup
