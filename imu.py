"""IMU serial driver for HI226/HI229 (or compatible) AHRS module.

Protocol parsing adapted from the hardware-tested ``imu_receiver.py``.

The runtime expects:

    * ``ang_vel``           — 3-vector, rad/s, in the robot root frame.
    * ``projected_gravity`` — 3-vector, unit-length ``R_world→root · [0,0,-1]``.
    * ``yaw_world``         — float, current world-frame yaw (rad).

Mounting note — the physical IMU is rotated 90° about Z relative to
``base_link``, so we apply ``R_base_from_imu`` to map IMU-frame
vectors into the robot root frame.  The remapping is:

    robot_x =  imu_y
    robot_y = -imu_x
    robot_z =  imu_z
"""
from __future__ import annotations

import logging
import struct
import numpy as np
import serial
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE


log = logging.getLogger("olaf.imu")

# IMU position relative to base_link origin, in base_link frame (metres).
# From olaf_robstride.urdf imu_joint <origin xyz="-0.095795 0 0.014"/>.
IMU_OFFSET_IN_BASE = np.array([-0.095795, 0.0, 0.014], dtype=np.float32)

# ---- Serial protocol constants (from imu_receiver.py) --------------------
FRAME_HEAD = "fc"
FRAME_END  = "fd"
TYPE_IMU   = "40"
TYPE_AHRS  = "41"
IMU_LEN    = "38"   # 56 bytes
AHRS_LEN   = "30"   # 48 bytes

# Default serial port for the CP2102 USB-UART bridge
DEFAULT_PORT = (
    "/dev/serial/by-id/"
    "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
)
DEFAULT_BAUD = 921_600


def lever_arm_velocity_correction(v_imu_root: np.ndarray,
                                  ang_vel_root: np.ndarray) -> np.ndarray:
    """Transport IMU-site linear velocity to base_link origin.

    All inputs expressed in the root frame.  ``v_base = v_imu − ω × r``.
    """
    return v_imu_root - np.cross(ang_vel_root, IMU_OFFSET_IN_BASE)


def _projected_gravity_from_quat(qw: float, qx: float,
                                 qy: float, qz: float) -> np.ndarray:
    """Compute R_world→body @ [0,0,-1] from a body-to-world quaternion.

    The third column of R(q) is R @ [0,0,1], so R^T @ [0,0,-1] = −col_2(R).
    """
    col2 = np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ], dtype=np.float64)
    return (-col2).astype(np.float32)


class Imu:
    # 90° Z-rotation: IMU frame → robot base_link frame.
    # robot_x = imu_y, robot_y = −imu_x, robot_z = imu_z.
    R_base_from_imu: np.ndarray = np.array(
        [[0, 1, 0],
         [-1, 0, 0],
         [0, 0, 1]], dtype=np.float32,
    )

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = 1.0,
    ) -> None:
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=EIGHTBITS,
            parity=PARITY_NONE,
            stopbits=STOPBITS_ONE,
            timeout=timeout,
        )
        log.info("IMU opened on %s @ %d baud", port, baudrate)

    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (ang_vel_root[3], projected_gravity_root[3], yaw_world).

        Blocks until one complete pair of TYPE_IMU + TYPE_AHRS frames is
        received.  At 921 600 baud the IMU streams well above 200 Hz, so
        this should return in < 10 ms normally.
        """
        got_imu = False
        got_ahrs = False

        # Raw parsed values (IMU frame, before axis remap)
        accel = np.zeros(3, dtype=np.float32)
        ang_vel_raw = np.zeros(3, dtype=np.float32)
        heading_raw = 0.0
        qw = qx = qy = qz = 0.0

        ser = self._serial
        while ser.is_open:
            # Read until frame header 0xFC
            b = ser.read(1)
            if len(b) == 0:
                continue
            if b.hex() != FRAME_HEAD:
                continue

            head_type = ser.read(1).hex()
            check_len = ser.read(1).hex()

            # Skip SN, CRC8, CRC16 (4 bytes)
            ser.read(4)

            # Read the data payload (length from header) regardless of type,
            # so the stream stays aligned even for frame types we don't use.
            data_len = int(check_len, 16)
            data = ser.read(data_len)
            if len(data) < data_len:
                continue

            if head_type == TYPE_IMU and check_len == IMU_LEN:
                imu_vals = struct.unpack("12f ii", data[:56])
                accel[0] = imu_vals[3]
                accel[1] = imu_vals[4]
                accel[2] = imu_vals[5]
                got_imu = True

            elif head_type == TYPE_AHRS and check_len == AHRS_LEN:
                ahrs = struct.unpack("10f ii", data[:48])
                ang_vel_raw[0] = ahrs[0]  # roll speed (IMU frame)
                ang_vel_raw[1] = ahrs[1]  # pitch speed (IMU frame)
                ang_vel_raw[2] = ahrs[2]  # heading speed (IMU frame)
                heading_raw = ahrs[5]
                qw = ahrs[6]
                qx = ahrs[7]
                qy = ahrs[8]
                qz = ahrs[9]
                got_ahrs = True

            if got_imu and got_ahrs:
                break

        # --- Transform from IMU frame to robot root frame ---

        # Angular velocity: rotate into robot frame
        ang_vel = self.R_base_from_imu @ ang_vel_raw

        # Projected gravity: compute in IMU frame from quaternion, then rotate
        proj_g_imu = _projected_gravity_from_quat(qw, qx, qy, qz)
        proj_g = self.R_base_from_imu @ proj_g_imu

        # Yaw is rotation about Z, which is the same axis in both frames
        yaw_w = heading_raw

        return ang_vel, proj_g, float(yaw_w)

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()
            log.info("IMU closed")