"""Motor driver for Robstride RS00/RS02/RS03.

All joints use the in-tree `lib/robstride` package (`RobstrideBus`),
which owns its own socketcan socket and implements the real Robstride wire
format (4×16-bit big-endian words + torque in the arbitration ID extra_data).
"""
from __future__ import annotations

import logging
import struct
import sys
import time
from pathlib import Path

import can
import numpy as np

from config import (
    CAN_FD_ENABLED, DEFAULT_JOINT_POS, MotorKind, MOTOR_TABLE,
    JOINT_ORDER, N_JOINTS,
)

# The robstride package lives under lib/robstride; add lib/ to path.
_RS_LIB_PATH = Path(__file__).resolve().parent / "lib"
if str(_RS_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(_RS_LIB_PATH))
from robstride import RobstrideBus, Motor, CommunicationType  # noqa: E402
from robstride.table import (  # noqa: E402
    MODEL_MIT_POSITION_TABLE,
    MODEL_MIT_VELOCITY_TABLE,
    MODEL_MIT_TORQUE_TABLE,
)

log = logging.getLogger("olaf.motors")

_RS_MODEL: dict[MotorKind, str] = {
    MotorKind.ROBSTRIDE_RS00: "rs-00",
    MotorKind.ROBSTRIDE_RS02: "rs-02",
    MotorKind.ROBSTRIDE_RS03: "rs-03",
}


class _FdBusWrapper:
    """Wraps a python-can Bus and promotes every outgoing frame to CAN-FD
    with bitrate-switch enabled. Inbound frames pass through untouched —
    the kernel socket already sees FD frames once the interface is
    configured with `fd on`. Lets us turn on CAN-FD without forking the
    robstride library.
    """

    def __init__(self, inner: "can.BusABC") -> None:
        self._inner = inner

    def send(self, msg: "can.Message", timeout: float | None = None) -> None:
        fd_msg = can.Message(
            arbitration_id=msg.arbitration_id,
            data=bytes(msg.data),
            is_extended_id=msg.is_extended_id,
            dlc=len(msg.data),
            is_fd=True,
            bitrate_switch=True,
        )
        self._inner.send(fd_msg, timeout=timeout)

    def recv(self, timeout: float | None = None):
        return self._inner.recv(timeout=timeout)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _open_fd_bus(channel: str) -> _FdBusWrapper:
    return _FdBusWrapper(can.interface.Bus(
        channel=channel, interface="socketcan", fd=True,
    ))


class MotorBus:
    """Coordinates Robstride motors across two CAN channels."""

    def __init__(self, can_usb: str = "can_usb", can_spi: str = "can_spi",
                 active_channels: tuple[str, ...] | None = None):
        self._specs = [MOTOR_TABLE[name] for name in JOINT_ORDER]

        # Route by can_id: 1-6 → can_usb, 7-12 → can_spi.
        def _channel_for(can_id: int) -> str:
            return can_usb if can_id <= 6 else can_spi

        channels = (can_usb, can_spi)
        if active_channels is None:
            active = set(channels)
        else:
            active = set(active_channels)
            unknown = active - set(channels)
            if unknown:
                raise ValueError(f"active_channels contains unknown {unknown}")
        self._active_channels = active

        rs_motors_by_ch: dict[str, dict[str, Motor]] = {c: {} for c in channels}
        rs_calib_by_ch: dict[str, dict[str, dict]] = {c: {} for c in channels}
        self._rs_names: dict[int, str] = {}
        self._rs_channel: dict[int, str] = {}
        self._rs_id_to_idx: dict[tuple[str, int], int] = {}
        self._live_idx: list[int] = []
        for i, s in enumerate(self._specs):
            name = JOINT_ORDER[i]
            ch = _channel_for(s.can_id)
            self._rs_names[i] = name
            self._rs_channel[i] = ch
            if ch not in active:
                continue
            rs_motors_by_ch[ch][name] = Motor(id=s.can_id, model=_RS_MODEL[s.kind])
            rs_calib_by_ch[ch][name] = {
                "direction": s.direction,
                "homing_offset": s.zero_offset_rad,
            }
            self._rs_id_to_idx[(ch, s.can_id)] = i
            self._live_idx.append(i)

        if not self._live_idx:
            raise RuntimeError("No active motors — check --bus selection")
        if active != set(channels):
            skipped = [JOINT_ORDER[i] for i in range(N_JOINTS)
                       if i not in self._live_idx]
            log.warning("SINGLE-BUS MODE: driving %s only; %d joints "
                        "pinned to DEFAULT_JOINT_POS: %s",
                        sorted(active), len(skipped), ", ".join(skipped))

        self._rs_libs: dict[str, RobstrideBus] = {}
        for ch in channels:
            if ch not in active or not rs_motors_by_ch[ch]:
                continue
            lib = RobstrideBus(ch, rs_motors_by_ch[ch],
                               calibration=rs_calib_by_ch[ch])
            lib.connect()
            if CAN_FD_ENABLED:
                lib.channel_handler.shutdown()
                lib.channel_handler = _open_fd_bus(ch)
                log.info("Robstride bus on %s promoted to CAN-FD", ch)
            self._rs_libs[ch] = lib

        # Latest feedback (URDF-frame, calibration already undone). Inactive
        # joints are pinned to DEFAULT_JOINT_POS so the policy observation
        # sees a sensible standing-pose value instead of zero.
        self._pos = DEFAULT_JOINT_POS.astype(np.float32).copy()
        self._vel = np.zeros(N_JOINTS, dtype=np.float32)
        self._tau = np.zeros(N_JOINTS, dtype=np.float32)

    # -- lifecycle ----------------------------------------------------------
    def enable_all(self) -> None:
        for i in self._live_idx:
            name = self._rs_names[i]
            try:
                self._rs_libs[self._rs_channel[i]].enable(name)
            except Exception as e:
                log.warning("RS enable failed for %s: %s", name, e)
            time.sleep(0.002)

    def disable_all(self) -> None:
        for i in self._live_idx:
            name = self._rs_names[i]
            try:
                self._rs_libs[self._rs_channel[i]].disable(name)
            except Exception as e:
                log.warning("RS disable failed for %s: %s", name, e)
        for lib in self._rs_libs.values():
            try:
                lib.disconnect(disable_torque=False)
            except Exception:
                pass

    # -- homing -------------------------------------------------------------
    def home(self, target: np.ndarray, speed: float = 0.3,
             dt: float = 0.01, timeout_s: float = 20.0,
             settle_pos_tol: float = 0.02,
             settle_vel_tol: float = 0.3) -> None:
        """Ramp every motor from its current position to ``target`` at
        ``speed`` rad/s (URDF frame).

        A power cycle can leave the Robstride multi-turn counter offset by
        an integer number of 2π; we pick the goal ≡ target (mod 2π) that
        is closest to the reported start so travel stays within ±π.
        """
        two_pi = 2.0 * float(np.pi)
        step = float(speed * dt)

        # Read start positions (URDF frame — calibration is applied by the lib).
        start_pos = target.astype(np.float32).copy()
        for i in self._live_idx:
            name = self._rs_names[i]
            lib = self._rs_libs[self._rs_channel[i]]
            # Zero-torque poke so a fresh status frame comes back.
            lib.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0)
            p, _, _, _ = lib.read_operation_frame(name)
            start_pos[i] = p

        goals = target.astype(np.float32).copy()
        for i in self._live_idx:
            k = round((start_pos[i] - target[i]) / two_pi)
            goals[i] = target[i] + k * two_pi

        travel = max((abs(start_pos[i] - goals[i]) for i in self._live_idx),
                     default=0.0)
        log.info("Homing %d motors: max travel %.3f rad @ %.2f rad/s "
                 "(est %.1f s)", len(self._live_idx), travel, speed,
                 travel / max(speed, 1e-6))

        targets = start_pos.copy()
        settled = {i: False for i in self._live_idx}
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            for i in self._live_idx:
                name = self._rs_names[i]
                diff = targets[i] - goals[i]
                if diff > step:
                    targets[i] -= step
                elif diff < -step:
                    targets[i] += step
                else:
                    targets[i] = goals[i]

                spec = self._specs[i]
                lib = self._rs_libs[self._rs_channel[i]]
                lib.write_operation_frame(
                    name, position=float(targets[i]),
                    kp=spec.kp, kd=spec.kd,
                )
                p, v, _, _ = lib.read_operation_frame(name)
                self._pos[i] = p
                self._vel[i] = v
                if (not settled[i]
                        and targets[i] == goals[i]
                        and abs(p - goals[i]) < settle_pos_tol
                        and abs(v) < settle_vel_tol):
                    settled[i] = True
                    log.info("homed: %s at %+.4f rad", name, p)

            if all(settled.values()):
                log.info("Homing complete")
                return
            time.sleep(dt)

        n_done = sum(1 for v in settled.values() if v)
        log.warning("Homing timed out — %d/%d motors settled",
                    n_done, len(self._live_idx))

    # -- command ------------------------------------------------------------
    def command(self, q_des: np.ndarray, qd_des: np.ndarray,
                kp_scale: float = 1.0) -> None:
        for i in self._live_idx:
            s = self._specs[i]
            q_cmd = float(q_des[i])
            qd_cmd = float(qd_des[i])
            kp = s.kp * kp_scale
            kd = s.kd
            # RobstrideBus.write_operation_frame applies calibration
            # (direction + homing_offset) internally — pass URDF-frame.
            self._rs_libs[self._rs_channel[i]].write_operation_frame(
                self._rs_names[i],
                position=q_cmd,
                kp=kp,
                kd=kd,
                velocity=qd_cmd,
            )

    def damp_all(self) -> None:
        for i in self._live_idx:
            s = self._specs[i]
            self._rs_libs[self._rs_channel[i]].write_operation_frame(
                self._rs_names[i],
                position=0.0, kp=0.0, kd=s.kd, velocity=0.0,
            )

    # -- feedback -----------------------------------------------------------
    def pump_feedback(self, budget_s: float = 0.001) -> None:
        """Drain pending CAN frames (non-blocking) and update joint state."""
        deadline = time.monotonic() + budget_s
        for ch_name, lib in self._rs_libs.items():
            handler = lib.channel_handler
            while handler is not None and time.monotonic() < deadline:
                frame = handler.recv(timeout=0.0)
                if frame is None:
                    break
                self._ingest_robstride(ch_name, frame)

    def _ingest_robstride(self, channel: str, frame) -> None:
        if not frame.is_extended_id or len(frame.data) < 8:
            return
        comm = (frame.arbitration_id >> 24) & 0x1F
        if comm != CommunicationType.OPERATION_STATUS:
            return
        device_id = (frame.arbitration_id >> 8) & 0xFF
        idx = self._rs_id_to_idx.get((channel, device_id))
        if idx is None:
            return
        spec = self._specs[idx]
        model = _RS_MODEL[spec.kind]
        pos_u16, vel_u16, torque_u16, _temp_u16 = struct.unpack(">HHHH",
                                                                frame.data)
        pos = (float(pos_u16) / 0x7FFF - 1.0) * MODEL_MIT_POSITION_TABLE[model]
        vel = (float(vel_u16) / 0x7FFF - 1.0) * MODEL_MIT_VELOCITY_TABLE[model]
        torque = (float(torque_u16) / 0x7FFF - 1.0) * MODEL_MIT_TORQUE_TABLE[model]
        self._pos[idx] = (pos - spec.zero_offset_rad) * spec.direction
        self._vel[idx] = vel * spec.direction
        self._tau[idx] = torque * spec.direction

    @property
    def joint_pos(self) -> np.ndarray:
        return self._pos

    @property
    def joint_vel(self) -> np.ndarray:
        return self._vel
