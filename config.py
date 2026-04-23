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
    0.00, -0.10,  0.90,  1.65,  0.75,  0.00,   # left leg
    0.00, -0.10, -0.90, -1.65,  0.75,  0.00,   # right leg
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
    ROBSTRIDE_RS00 = "rs00"   # ankle pitch/roll
    ROBSTRIDE_RS02 = "rs02"   # hip yaw
    ROBSTRIDE_RS03 = "rs03"   # hip roll/pitch, knee


@dataclass(frozen=True)
class MotorSpec:
    can_id: int             # arbitration ID on the bus
    kind: MotorKind
    direction: int = 1      # +1 or -1 to flip sign vs. URDF convention
    zero_offset_rad: float = 0.0  # mechanical zero offset (encoder − URDF zero)
    kp: float = 0.0
    kd: float = 0.0


# ---------------------------------------------------------------------------
# Motor wiring — IDs match the tested all_homing.py layout.
# ---------------------------------------------------------------------------
MOTOR_TABLE: dict[str, MotorSpec] = {
    "l_hip_yaw_joint":     MotorSpec(can_id=1,  kind=MotorKind.ROBSTRIDE_RS02, kp=18.0, kd=1.5),
    "l_hip_roll_joint":    MotorSpec(can_id=2,  kind=MotorKind.ROBSTRIDE_RS03, kp=45.0, kd=6.0),
    "l_hip_pitch_joint":   MotorSpec(can_id=3,  kind=MotorKind.ROBSTRIDE_RS03, kp=45.0, kd=6.0),
    "l_knee_pitch_joint":  MotorSpec(can_id=4,  kind=MotorKind.ROBSTRIDE_RS03, direction=-1, kp=45.0, kd=6.0),
    "l_ankle_pitch_joint": MotorSpec(can_id=5,  kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=30.0, kd=4.0),
    "l_ankle_roll_joint":  MotorSpec(can_id=6,  kind=MotorKind.ROBSTRIDE_RS00, kp=15.0, kd=1.5),

    # Right-leg motors are physically mirrored from the left, so every
    # right actuator is flipped (direction=-1) to match URDF sign convention.
    "r_hip_yaw_joint":     MotorSpec(can_id=7,  kind=MotorKind.ROBSTRIDE_RS02, direction=-1, kp=18.0, kd=1.5),
    "r_hip_roll_joint":    MotorSpec(can_id=8,  kind=MotorKind.ROBSTRIDE_RS03, direction=1, kp=45.0, kd=6.0),
    "r_hip_pitch_joint":   MotorSpec(can_id=9,  kind=MotorKind.ROBSTRIDE_RS03, direction=-1, kp=45.0, kd=6.0),
    "r_knee_pitch_joint":  MotorSpec(can_id=10, kind=MotorKind.ROBSTRIDE_RS03, direction=1, kp=45.0, kd=6.0),
    "r_ankle_pitch_joint": MotorSpec(can_id=11, kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=30.0, kd=4.0),
    "r_ankle_roll_joint":  MotorSpec(can_id=12, kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=15.0, kd=1.5),
}

# ---------------------------------------------------------------------------
# Loop rates
# ---------------------------------------------------------------------------
POLICY_HZ       = 50.0
# 200 Hz gives 5× Nyquist headroom over the 37.5 Hz LPF cutoff and keeps
# total CAN traffic (12 motors × (cmd + reply)) under the 1 Mbps bus.
# 600 Hz saturated the bus once the Robstride library started receiving
# real status replies from the motors.
MOTOR_HZ        = 200.0
POLICY_DT       = 1.0 / POLICY_HZ
MOTOR_DT        = 1.0 / MOTOR_HZ
LPF_CUTOFF_HZ   = 37.5                      # per paper §VII-C
GAIT_PERIOD_S   = 0.6                       # must match ObsTerm gait_phase params

# Safety
POLICY_WATCHDOG_S = 0.040                   # drop to damping if policy misses
KP_RAMP_S         = 2.0                     # soft-start ramp on startup

# Slow-motion debug: per-joint velocity cap applied to the policy target
# before it's published to the motor thread. Enabled via `run.py --slomo`.
SLOMO_VMAX_RAD_S  = 0.5

# ---------------------------------------------------------------------------
# CAN-FD
# ---------------------------------------------------------------------------
# When True, all outgoing frames (Robstride + Damiao) are promoted to CAN-FD
# with bitrate-switch enabled. Requires THREE things configured elsewhere:
#   1. Kernel interface in FD mode:
#        sudo ip link set can_usb down
#        sudo ip link set can_usb type can bitrate 1000000 dbitrate 5000000 fd on
#        sudo ip link set can_usb up txqueuelen 1000
#   2. Each motor's firmware configured for CAN-FD at the chosen data bitrate
#      (Robstride + Damiao both support it, but must be enabled per-motor).
#   3. A CAN adapter that supports FD (CANable at 5 Mbps is fine).
# Leave False until all three are true — otherwise every send raises.
CAN_FD_ENABLED    = False
CAN_FD_DBITRATE   = 5_000_000
