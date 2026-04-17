"""Motor drivers for Robstride RS02/RS03 and Damiao DM-J43xx.

Robstride motors go through the in-tree `canable/robstride/lib` package
(`RobstrideBus`), which owns its own socketcan socket and implements the
real Robstride wire format (4×16-bit big-endian words + torque in the
arbitration ID extra_data). Damiao motors keep the existing
``damiao-motor`` library path.
"""
from __future__ import annotations

import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np
from damiao_motor import DaMiaoController

from can_bus import CanBus
from config import MotorKind, MOTOR_TABLE, JOINT_ORDER, N_JOINTS

# The robstride package lives under canable/robstride/lib; add it to path.
_RS_LIB_PATH = Path(__file__).resolve().parent / "canable" / "robstride" / "lib"
if str(_RS_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(_RS_LIB_PATH))
from robstride import RobstrideBus, Motor, CommunicationType  # noqa: E402
from robstride.table import (  # noqa: E402
    MODEL_MIT_POSITION_TABLE,
    MODEL_MIT_VELOCITY_TABLE,
    MODEL_MIT_TORQUE_TABLE,
)

log = logging.getLogger("olaf.motors")

_RS_KINDS = (MotorKind.ROBSTRIDE_RS02, MotorKind.ROBSTRIDE_RS03)
_DM_KINDS = (MotorKind.DAMIAO_J4340P, MotorKind.DAMIAO_J4310)

_RS_MODEL: dict[MotorKind, str] = {
    MotorKind.ROBSTRIDE_RS02: "rs-02",
    MotorKind.ROBSTRIDE_RS03: "rs-03",
}


class MotorBus:
    """Coordinates Robstride (via RobstrideBus) and Damiao (via damiao-motor)
    behind a single vectorised interface.
    """

    def __init__(self, rs_bus: CanBus, dm_channel: str = "can_usb"):
        # rs_bus is no longer used for Robstride (the library owns its own
        # socket) but we keep the arg so run.py doesn't need to change.
        del rs_bus
        self._specs = [MOTOR_TABLE[name] for name in JOINT_ORDER]

        # --- Robstride via RobstrideBus ---
        rs_motors: dict[str, Motor] = {}
        rs_calibration: dict[str, dict] = {}
        self._rs_names: dict[int, str] = {}
        self._rs_id_to_idx: dict[int, int] = {}
        for i, s in enumerate(self._specs):
            if s.kind in _RS_KINDS:
                name = JOINT_ORDER[i]
                rs_motors[name] = Motor(id=s.can_id, model=_RS_MODEL[s.kind])
                rs_calibration[name] = {
                    "direction": s.direction,
                    "homing_offset": s.zero_offset_rad,
                }
                self._rs_names[i] = name
                self._rs_id_to_idx[s.can_id] = i
        self._rs_lib = RobstrideBus(dm_channel, rs_motors,
                                    calibration=rs_calibration)
        self._rs_lib.connect()

        # --- Damiao via damiao-motor ---
        self._dm_ctrl = DaMiaoController(channel=dm_channel, bustype="socketcan")
        self._dm_motors: dict[int, object] = {}
        for i, s in enumerate(self._specs):
            if s.kind in _DM_KINDS:
                motor = self._dm_ctrl.add_motor(
                    motor_id=s.can_id,
                    feedback_id=s.master_id,
                    motor_type=s.dm_type,
                )
                self._dm_motors[i] = motor

        # Latest feedback (URDF-frame, calibration already undone)
        self._pos = np.zeros(N_JOINTS, dtype=np.float32)
        self._vel = np.zeros(N_JOINTS, dtype=np.float32)
        self._tau = np.zeros(N_JOINTS, dtype=np.float32)

    # -- lifecycle ----------------------------------------------------------
    def enable_all(self) -> None:
        for name in self._rs_names.values():
            try:
                self._rs_lib.enable(name)
            except Exception as e:
                log.warning("RS enable failed for %s: %s", name, e)
            time.sleep(0.002)
        for motor in self._dm_motors.values():
            motor.enable()
            time.sleep(0.002)

    def disable_all(self) -> None:
        for name in self._rs_names.values():
            try:
                self._rs_lib.disable(name)
            except Exception as e:
                log.warning("RS disable failed for %s: %s", name, e)
        for motor in self._dm_motors.values():
            try:
                motor.disable()
            except Exception:
                pass
        try:
            self._rs_lib.disconnect(disable_torque=False)
        except Exception:
            pass

    # -- command ------------------------------------------------------------
    def command(self, q_des: np.ndarray, qd_des: np.ndarray,
                kp_scale: float = 1.0) -> None:
        for i, s in enumerate(self._specs):
            q_cmd = float(q_des[i])
            qd_cmd = float(qd_des[i])
            kp = s.kp * kp_scale
            kd = s.kd

            if s.kind in _RS_KINDS:
                # RobstrideBus.write_operation_frame applies calibration
                # (direction + homing_offset) internally — pass URDF-frame.
                self._rs_lib.write_operation_frame(
                    self._rs_names[i],
                    position=q_cmd,
                    kp=kp,
                    kd=kd,
                    velocity=qd_cmd,
                )
            else:
                # Damiao takes motor-frame, so apply calibration here.
                dm_q  = q_cmd  * s.direction + s.zero_offset_rad
                dm_qd = qd_cmd * s.direction
                self._dm_motors[i].send_cmd_mit(
                    target_position=dm_q,
                    target_velocity=dm_qd,
                    stiffness=kp,
                    damping=kd,
                    feedforward_torque=0.0,
                )

    def damp_all(self) -> None:
        for i, s in enumerate(self._specs):
            if s.kind in _RS_KINDS:
                self._rs_lib.write_operation_frame(
                    self._rs_names[i],
                    position=0.0, kp=0.0, kd=s.kd, velocity=0.0,
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
        """Drain pending CAN frames (non-blocking) and update joint state."""
        deadline = time.monotonic() + budget_s
        ch = self._rs_lib.channel_handler
        while ch is not None and time.monotonic() < deadline:
            frame = ch.recv(timeout=0.0)
            if frame is None:
                break
            self._ingest_robstride(frame)

        # Damiao: library has a background poller; just copy latest state.
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

    def _ingest_robstride(self, frame) -> None:
        if not frame.is_extended_id or len(frame.data) < 8:
            return
        comm = (frame.arbitration_id >> 24) & 0x1F
        if comm != CommunicationType.OPERATION_STATUS:
            return
        device_id = (frame.arbitration_id >> 8) & 0xFF
        idx = self._rs_id_to_idx.get(device_id)
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
