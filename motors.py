"""MIT-mode CAN drivers for Robstride RS02/RS03 and Damiao DM-J43xx.

Both families use the MIT impedance command:

    τ = Kp·(q_des − q) + Kd·(q̇_des − q̇) + τ_ff

packed into 8 CAN bytes.  Per-family constants (ID format, scaling ranges)
differ and are in ``_SPEC`` tables below.  **Verify these against your
firmware manual before enabling torque** — vendors ship revisions that
change the ranges.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np

from .can_bus import CanBus, CanFrame
from .config import MotorKind, MotorSpec, MOTOR_TABLE, JOINT_ORDER, N_JOINTS


# ---------------------------------------------------------------------------
# Per-family packing ranges  (pos_rad, vel_rad_s, kp, kd, tau_Nm)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Spec:
    p_lim: float
    v_lim: float
    kp_lim: float
    kd_lim: float
    t_lim: float


_SPEC: dict[MotorKind, _Spec] = {
    # Robstride uses the "CyberGear"-style ranges.  RS02/RS03 share the
    # position/kp/kd ranges; velocity and torque differ by model.
    MotorKind.ROBSTRIDE_RS02: _Spec(p_lim=12.5, v_lim=42.0, kp_lim=500.0, kd_lim=5.0, t_lim=17.0),
    MotorKind.ROBSTRIDE_RS03: _Spec(p_lim=12.5, v_lim=20.4, kp_lim=500.0, kd_lim=5.0, t_lim=60.0),
    # Damiao default MIT ranges.  Check the DIP-switch / RID config on
    # each motor — some shipments come with p_lim=3.14.
    MotorKind.DAMIAO_J4340:   _Spec(p_lim=12.5, v_lim= 8.0, kp_lim=500.0, kd_lim=5.0, t_lim=27.0),
    MotorKind.DAMIAO_J4310:   _Spec(p_lim=12.5, v_lim=30.0, kp_lim=500.0, kd_lim=5.0, t_lim=7.0),
}


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------
def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(min(x, x_max), x_min)
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)


def _uint_to_float(u: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    return u * span / ((1 << bits) - 1) + x_min


def _pack_mit(pos: float, vel: float, kp: float, kd: float, tau_ff: float,
              spec: _Spec) -> bytes:
    """Pack MIT command: pos 16b | vel 12b | kp 12b | kd 12b | tau 12b."""
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


def _unpack_mit_reply(data: bytes, spec: _Spec) -> tuple[float, float, float]:
    """Decode (pos, vel, tau) from an MIT reply (Damiao/Robstride share layout)."""
    p = (data[1] << 8) | data[2]
    v = (data[3] << 4) | (data[4] >> 4)
    t = ((data[4] & 0x0F) << 8) | data[5]
    return (
        _uint_to_float(p, -spec.p_lim, +spec.p_lim, 16),
        _uint_to_float(v, -spec.v_lim, +spec.v_lim, 12),
        _uint_to_float(t, -spec.t_lim, +spec.t_lim, 12),
    )


# ---------------------------------------------------------------------------
# Frame builders — family-specific IDs
# ---------------------------------------------------------------------------
def _robstride_frame(motor_id: int, payload: bytes, cmd_type: int) -> CanFrame:
    """Robstride 29-bit extended ID: [type:5][0:16][host:8][motor:8]."""
    host_id = 0xFD
    arb = (cmd_type & 0x1F) << 24 | (host_id & 0xFF) << 8 | (motor_id & 0xFF)
    return CanFrame(arbitration_id=arb, data=payload, is_extended_id=True)


def _damiao_frame(motor_id: int, payload: bytes) -> CanFrame:
    """Damiao: 11-bit standard ID equal to the motor's CAN ID."""
    return CanFrame(arbitration_id=motor_id, data=payload, is_extended_id=False)


def _build_command(spec_m: MotorSpec, q: float, qd: float,
                   kp: float, kd: float, tau_ff: float) -> CanFrame:
    spec = _SPEC[spec_m.kind]
    q_cmd  = q  * spec_m.direction + spec_m.zero_offset_rad
    qd_cmd = qd * spec_m.direction
    payload = _pack_mit(q_cmd, qd_cmd, kp, kd, tau_ff, spec)
    if spec_m.kind in (MotorKind.ROBSTRIDE_RS02, MotorKind.ROBSTRIDE_RS03):
        return _robstride_frame(spec_m.can_id, payload, cmd_type=0x01)
    return _damiao_frame(spec_m.can_id, payload)


def _enable_frame(spec_m: MotorSpec) -> CanFrame:
    if spec_m.kind in (MotorKind.ROBSTRIDE_RS02, MotorKind.ROBSTRIDE_RS03):
        # Robstride enable: cmd type 0x03, empty payload
        return _robstride_frame(spec_m.can_id, bytes(8), cmd_type=0x03)
    # Damiao enable: 0xFF * 7 + 0xFC
    return _damiao_frame(spec_m.can_id, bytes([0xFF] * 7 + [0xFC]))


def _disable_frame(spec_m: MotorSpec) -> CanFrame:
    if spec_m.kind in (MotorKind.ROBSTRIDE_RS02, MotorKind.ROBSTRIDE_RS03):
        return _robstride_frame(spec_m.can_id, bytes(8), cmd_type=0x04)
    return _damiao_frame(spec_m.can_id, bytes([0xFF] * 7 + [0xFD]))


# ---------------------------------------------------------------------------
# MotorBus — vectorised interface used by run.py
# ---------------------------------------------------------------------------
class MotorBus:
    """Thin coordinator — sends N_JOINTS commands per tick, reads replies
    whenever the bus is drained (best-effort, non-blocking).
    """

    def __init__(self, bus: CanBus):
        self._bus = bus
        self._specs = [MOTOR_TABLE[name] for name in JOINT_ORDER]
        self._id_to_idx = {s.can_id: i for i, s in enumerate(self._specs)}
        # Latest feedback (URDF-frame, after undoing direction/offset)
        self._pos = np.zeros(N_JOINTS, dtype=np.float32)
        self._vel = np.zeros(N_JOINTS, dtype=np.float32)
        self._tau = np.zeros(N_JOINTS, dtype=np.float32)

    # -- lifecycle ----------------------------------------------------------
    def enable_all(self) -> None:
        for s in self._specs:
            self._bus.send(_enable_frame(s))
            time.sleep(0.002)

    def disable_all(self) -> None:
        for s in self._specs:
            try:
                self._bus.send(_disable_frame(s))
            except Exception:
                pass

    # -- command ------------------------------------------------------------
    def command(self, q_des: np.ndarray, qd_des: np.ndarray,
                kp_scale: float = 1.0) -> None:
        """Send one MIT command per joint.  ``q_des`` / ``qd_des`` are in
        URDF frame and JOINT_ORDER order.  ``kp_scale`` in [0,1] lets the
        caller soft-start (ramp from 0 → 1).
        """
        for i, s in enumerate(self._specs):
            frame = _build_command(
                s,
                q=float(q_des[i]),
                qd=float(qd_des[i]),
                kp=s.kp * kp_scale,
                kd=s.kd,
                tau_ff=0.0,
            )
            self._bus.send(frame)

    def damp_all(self) -> None:
        """Kp=0, Kd=nominal — safe fallback pose."""
        for s in self._specs:
            frame = _build_command(s, q=0.0, qd=0.0, kp=0.0, kd=s.kd, tau_ff=0.0)
            self._bus.send(frame)

    # -- feedback -----------------------------------------------------------
    def pump_feedback(self, budget_s: float = 0.001) -> None:
        """Drain pending CAN frames for up to ``budget_s`` and update state."""
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            frame = self._bus.recv(timeout=0.0)
            if frame is None:
                return
            self._ingest(frame)

    def _ingest(self, frame: CanFrame) -> None:
        # Robstride replies carry motor_id in byte[1] of the ID; Damiao in
        # arbitration_id itself.  This is intentionally permissive — if your
        # adapter strips extended-ID metadata, adjust here.
        if frame.is_extended_id:
            motor_id = (frame.arbitration_id >> 8) & 0xFF  # Robstride
        else:
            # Damiao replies arrive on (motor_id + 0x10) by default
            motor_id = frame.arbitration_id & 0x0F
        idx = self._id_to_idx.get(motor_id)
        if idx is None or len(frame.data) < 6:
            return
        spec_m = self._specs[idx]
        spec = _SPEC[spec_m.kind]
        p, v, t = _unpack_mit_reply(frame.data, spec)
        # Undo direction flip and zero offset → URDF frame
        self._pos[idx] = (p - spec_m.zero_offset_rad) * spec_m.direction
        self._vel[idx] = v * spec_m.direction
        self._tau[idx] = t * spec_m.direction

    @property
    def joint_pos(self) -> np.ndarray:
        return self._pos

    @property
    def joint_vel(self) -> np.ndarray:
        return self._vel
