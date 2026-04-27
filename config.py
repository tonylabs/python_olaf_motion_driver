"""Wiring / kinematics constants for the physical OLAF.

Keep in sync with ``olaf_walking_env_cfg.py``.  The ORDER of ``JOINT_ORDER``
is the contract between policy outputs, observations, and motor commands.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np


# ---------------------------------------------------------------------------
# Joint order — MUST match Isaac Lab's `robot.data.joint_names`, which
# interleaves [L, R] per family. See SDK_DEPLOYMENT.md §2.
# ---------------------------------------------------------------------------
JOINT_ORDER = (
    "l_hip_yaw_joint",
    "r_hip_yaw_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)
N_JOINTS = len(JOINT_ORDER)

# Default joint pose `q0` (the bent-knee stance the policy was trained
# around). q_target = q0 + 0.5 * action. SDK_DEPLOYMENT.md §3 — L and R
# share numeric values; URDF axis flip handles the physical mirror, do
# NOT sign-flip targets here.
DEFAULT_JOINT_POS = np.array([
    0.000, 0.000,   # hip_yaw    (L, R)
    0.000, 0.000,   # hip_roll   (L, R)
    0.900, 0.900,   # hip_pitch  (L, R)
    1.650, 1.650,   # knee_pitch (L, R)
    0.700, 0.700,   # ankle_pitch(L, R)
    0.000, 0.000,   # ankle_roll (L, R)
], dtype=np.float32)

# URDF/MJCF joint limits (lower, upper) per SDK_DEPLOYMENT.md §7. Symmetric
# L/R; clamp q_target to these (NOT the action). Source: olaf_robstride.xml.
JOINT_LIMITS = np.array([
    (-0.2600,  0.2600),  # l_hip_yaw
    (-0.2600,  0.2600),  # r_hip_yaw
    (-0.2618,  0.3490),  # l_hip_roll
    (-0.2618,  0.3490),  # r_hip_roll
    (-0.1000,  1.4000),  # l_hip_pitch
    (-0.1000,  1.4000),  # r_hip_pitch
    (-0.4000,  2.0000),  # l_knee_pitch
    (-0.4000,  2.0000),  # r_knee_pitch
    (-0.6500,  1.0500),  # l_ankle_pitch
    (-0.6500,  1.0500),  # r_ankle_pitch
    (-0.2000,  0.2600),  # l_ankle_roll
    (-0.2000,  0.2600),  # r_ankle_roll
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
#
# kp/kd per actuator family come from SDK_DEPLOYMENT.md §2 (the trained
# gains: RS02 40/3.000, RS03 78.957/5.027, RS00 16.581/1.056). The inner
# PD must reproduce these so the policy doesn't see a different plant
# than it trained against.
#
# WARNING: the `direction` flags below were calibrated against the OLD
# asymmetric `DEFAULT_JOINT_POS` (which sign-flipped right-leg targets in
# URDF frame). The constants above now match SDK_DEPLOYMENT.md §3 (symmetric
# L/R values, mirroring lives in URDF axes). If the right leg drives in the
# wrong direction at boot ramp, the right-side `direction` calibrations
# (esp. r_hip_pitch_joint, r_knee_pitch_joint) likely need to be re-verified
# on bench — single-joint commands per §11 checklist.
# ---------------------------------------------------------------------------
_RS02_KP, _RS02_KD = 40.000, 3.000   # hip_yaw
_RS03_KP, _RS03_KD = 78.957, 5.027   # hip_roll, hip_pitch, knee_pitch
_RS00_KP, _RS00_KD = 16.581, 1.056   # ankle_pitch, ankle_roll

MOTOR_TABLE: dict[str, MotorSpec] = {
    "l_hip_yaw_joint":     MotorSpec(can_id=1,  kind=MotorKind.ROBSTRIDE_RS02, kp=_RS02_KP, kd=_RS02_KD),
    "l_hip_roll_joint":    MotorSpec(can_id=2,  kind=MotorKind.ROBSTRIDE_RS03, kp=_RS03_KP, kd=_RS03_KD),
    "l_hip_pitch_joint":   MotorSpec(can_id=3,  kind=MotorKind.ROBSTRIDE_RS03, kp=_RS03_KP, kd=_RS03_KD),
    "l_knee_pitch_joint":  MotorSpec(can_id=4,  kind=MotorKind.ROBSTRIDE_RS03, direction=-1, kp=_RS03_KP, kd=_RS03_KD),
    "l_ankle_pitch_joint": MotorSpec(can_id=5,  kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=_RS00_KP, kd=_RS00_KD),
    "l_ankle_roll_joint":  MotorSpec(can_id=6,  kind=MotorKind.ROBSTRIDE_RS00, kp=_RS00_KP, kd=_RS00_KD),

    # Right-leg motors are physically mirrored from the left, so every
    # right actuator is flipped (direction=-1) to match URDF sign convention.
    "r_hip_yaw_joint":     MotorSpec(can_id=7,  kind=MotorKind.ROBSTRIDE_RS02, direction=-1, kp=_RS02_KP, kd=_RS02_KD),
    "r_hip_roll_joint":    MotorSpec(can_id=8,  kind=MotorKind.ROBSTRIDE_RS03, direction=1,  kp=_RS03_KP, kd=_RS03_KD),
    "r_hip_pitch_joint":   MotorSpec(can_id=9,  kind=MotorKind.ROBSTRIDE_RS03, direction=1,  kp=_RS03_KP, kd=_RS03_KD),
    "r_knee_pitch_joint":  MotorSpec(can_id=10, kind=MotorKind.ROBSTRIDE_RS03, direction=-1, kp=_RS03_KP, kd=_RS03_KD),
    "r_ankle_pitch_joint": MotorSpec(can_id=11, kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=_RS00_KP, kd=_RS00_KD),
    "r_ankle_roll_joint":  MotorSpec(can_id=12, kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=_RS00_KP, kd=_RS00_KD),
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
