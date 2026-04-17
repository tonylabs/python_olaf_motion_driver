"""Motor drivers for Robstride RS02/RS03 and Damiao DM-J43xx.

Robstride motors use raw MIT-mode CAN frames via CanBus.
Damiao motors use the ``damiao-motor`` library (``DaMiaoController``),
which manages its own CAN bus connection and background feedback polling.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from damiao_motor import DaMiaoController

from can_bus import CanBus, CanFrame
from config import MotorKind, MotorSpec, MOTOR_TABLE, JOINT_ORDER, N_JOINTS

log = logging.getLogger("olaf.motors")

# Robstride motor kinds (use raw CAN MIT frames)
_RS_KINDS = (MotorKind.ROBSTRIDE_RS02, MotorKind.ROBSTRIDE_RS03)
# Damiao motor kinds (use damiao-motor library)
_DM_KINDS = (MotorKind.DAMIAO_J4340P, MotorKind.DAMIAO_J4310)


# ---------------------------------------------------------------------------
# Robstride MIT-mode helpers (unchanged from original)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _RobstrideSpec:
    p_lim: float
    v_lim: float
    kp_lim: float
    kd_lim: float
    t_lim: float


_RS_SPEC: dict[MotorKind, _RobstrideSpec] = {
    MotorKind.ROBSTRIDE_RS02: _RobstrideSpec(p_lim=12.5, v_lim=42.0, kp_lim=500.0, kd_lim=5.0, t_lim=17.0),
    MotorKind.ROBSTRIDE_RS03: _RobstrideSpec(p_lim=12.5, v_lim=20.4, kp_lim=500.0, kd_lim=5.0, t_lim=60.0),
}


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(min(x, x_max), x_min)
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)


def _uint_to_float(u: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    return u * span / ((1 << bits) - 1) + x_min


def _pack_mit(pos: float, vel: float, kp: float, kd: float, tau_ff: float,
              spec: _RobstrideSpec) -> bytes:
    p = _float_to_uint(pos,   -spec.p_lim, +spec.p_lim, 16)
    v = _float_to_uint(vel,   -spec.v_lim, +spec.v_lim, 12)
    k = _float_to_uint(kp,     0.0,        spec.kp_lim, 12)
    d = _float_to_uint(kd,     0.0,        spec.kd_lim, 12)
    t = _float_to_uint(tau_ff,-spec.t_lim, +spec.t_lim, 12)
    return bytes([
        (p >> 8) & 0xFF,
        p & 0xFF,
        (v >> 4) & 0xFF,
        ((v & 0x0F) << 4) | ((k >> 8) & 0x0F),
        k & 0xFF,
        (d >> 4) & 0xFF,
        ((d & 0x0F) << 4) | ((t >> 8) & 0x0F),
        t & 0xFF,
    ])


def _unpack_mit_reply(data: bytes, spec: _RobstrideSpec) -> tuple[float, float, float]:
    p = (data[1] << 8) | data[2]
    v = (data[3] << 4) | (data[4] >> 4)
    t = ((data[4] & 0x0F) << 8) | data[5]
    return (
        _uint_to_float(p, -spec.p_lim, +spec.p_lim, 16),
        _uint_to_float(v, -spec.v_lim, +spec.v_lim, 12),
        _uint_to_float(t, -spec.t_lim, +spec.t_lim, 12),
    )


def _robstride_frame(motor_id: int, payload: bytes, cmd_type: int) -> CanFrame:
    host_id = 0xFD
    arb = (cmd_type & 0x1F) << 24 | (host_id & 0xFF) << 8 | (motor_id & 0xFF)
    return CanFrame(arbitration_id=arb, data=payload, is_extended_id=True)


# ---------------------------------------------------------------------------
# MotorBus — vectorised interface used by run.py
# ---------------------------------------------------------------------------
class MotorBus:
    """Coordinates Robstride (raw CAN) and Damiao (damiao-motor library)
    motors behind a single vectorised interface.
    """

    def __init__(self, rs_bus: CanBus, dm_channel: str = "can_usb"):
        self._rs_bus = rs_bus
        self._specs = [MOTOR_TABLE[name] for name in JOINT_ORDER]

        # --- Robstride bookkeeping ---
        self._rs_id_to_idx: dict[int, int] = {}
        for i, s in enumerate(self._specs):
            if s.kind in _RS_KINDS:
                self._rs_id_to_idx[s.can_id] = i

        # --- Damiao: set up via damiao-motor library ---
        self._dm_ctrl = DaMiaoController(channel=dm_channel, bustype="socketcan")
        self._dm_motors: dict[int, object] = {}   # idx → DaMiaoMotor
        for i, s in enumerate(self._specs):
            if s.kind in _DM_KINDS:
                motor = self._dm_ctrl.add_motor(
                    motor_id=s.can_id,
                    master_id=s.master_id,
                    motor_type=s.dm_type,
                )
                self._dm_motors[i] = motor

        # Latest feedback (URDF-frame, after undoing direction/offset)
        self._pos = np.zeros(N_JOINTS, dtype=np.float32)
        self._vel = np.zeros(N_JOINTS, dtype=np.float32)
        self._tau = np.zeros(N_JOINTS, dtype=np.float32)

    # -- lifecycle ----------------------------------------------------------
    def enable_all(self) -> None:
        # Robstride: send enable frame
        for s in self._specs:
            if s.kind in _RS_KINDS:
                frame = _robstride_frame(s.can_id, bytes(8), cmd_type=0x03)
                self._rs_bus.send(frame)
                time.sleep(0.002)
        # Damiao: enable via library
        for motor in self._dm_motors.values():
            motor.enable()
            time.sleep(0.002)

    def disable_all(self) -> None:
        for s in self._specs:
            if s.kind in _RS_KINDS:
                try:
                    frame = _robstride_frame(s.can_id, bytes(8), cmd_type=0x04)
                    self._rs_bus.send(frame)
                except Exception:
                    pass
        for motor in self._dm_motors.values():
            try:
                motor.disable()
            except Exception:
                pass

    # -- command ------------------------------------------------------------
    def command(self, q_des: np.ndarray, qd_des: np.ndarray,
                kp_scale: float = 1.0) -> None:
        for i, s in enumerate(self._specs):
            q_cmd = float(q_des[i]) * s.direction + s.zero_offset_rad
            qd_cmd = float(qd_des[i]) * s.direction
            kp = s.kp * kp_scale
            kd = s.kd

            if s.kind in _RS_KINDS:
                spec = _RS_SPEC[s.kind]
                payload = _pack_mit(q_cmd, qd_cmd, kp, kd, 0.0, spec)
                self._rs_bus.send(
                    _robstride_frame(s.can_id, payload, cmd_type=0x01)
                )
            else:
                self._dm_motors[i].send_cmd_mit(
                    target_position=q_cmd,
                    target_velocity=qd_cmd,
                    stiffness=kp,
                    damping=kd,
                    feedforward_torque=0.0,
                )

    def damp_all(self) -> None:
        for i, s in enumerate(self._specs):
            if s.kind in _RS_KINDS:
                spec = _RS_SPEC[s.kind]
                payload = _pack_mit(0.0, 0.0, 0.0, s.kd, 0.0, spec)
                self._rs_bus.send(
                    _robstride_frame(s.can_id, payload, cmd_type=0x01)
                )
            else:
                self._dm_motors[i].send_cmd_mit(
                    target_position=0.0,
                    target_velocity=0.0,
                    stiffness=0.0,
                    damping=s.kd,
                    feedforward_torque=0.0,
                )

    # -- feedback -----------------------------------------------------------
    def pump_feedback(self, budget_s: float = 0.001) -> None:
        """Drain pending CAN frames and update state."""
        # Robstride: drain from raw bus
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            frame = self._rs_bus.recv(timeout=0.0)
            if frame is None:
                break
            self._ingest_robstride(frame)

        # Damiao: read latest state from library (background thread polls)
        self._dm_ctrl.poll_feedback()
        for idx, motor in self._dm_motors.items():
            spec_m = self._specs[idx]
            state = motor.state
            p = state.get("pos", 0.0)
            v = state.get("vel", 0.0)
            t = state.get("torq", 0.0)
            self._pos[idx] = (p - spec_m.zero_offset_rad) * spec_m.direction
            self._vel[idx] = v * spec_m.direction
            self._tau[idx] = t * spec_m.direction

    def _ingest_robstride(self, frame: CanFrame) -> None:
        if not frame.is_extended_id:
            return
        motor_id = (frame.arbitration_id >> 8) & 0xFF
        idx = self._rs_id_to_idx.get(motor_id)
        if idx is None or len(frame.data) < 6:
            return
        spec_m = self._specs[idx]
        spec = _RS_SPEC[spec_m.kind]
        p, v, t = _unpack_mit_reply(frame.data, spec)
        self._pos[idx] = (p - spec_m.zero_offset_rad) * spec_m.direction
        self._vel[idx] = v * spec_m.direction
        self._tau[idx] = t * spec_m.direction

    @property
    def joint_pos(self) -> np.ndarray:
        return self._pos

    @property
    def joint_vel(self) -> np.ndarray:
        return self._vel
