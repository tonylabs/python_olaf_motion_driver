"""IMU interface stub.

Replace with your sensor driver (BMI088, ICM-20948, VectorNav, etc.).  The
runtime expects:

    * ``ang_vel``           — 3-vector, rad/s, **expressed in the robot
                              root frame** (URDF ``base_link``).
    * ``projected_gravity`` — 3-vector, unit-length ``R_world→root · [0,0,-1]``.
    * ``yaw_world``         — float, current world-frame yaw (rad), used by
                              the path-frame tracker.

Mounting per URDF (``olaf_robstride.urdf``, ``imu_joint``):

        origin xyz = (-0.095795, 0.0, 0.014)   # metres, in base_link
        origin rpy = (0, 0, 0)                 # axes identity with base_link

Implications:

    * Rotation is identity, so no ``R_base←imu`` is needed **if your
      physical sensor is mounted in the orientation the URDF assumes**.
      Verify on the bench: tilt the robot +5° pitch forward, the IMU
      should read +5° pitch, not roll.  If mismatched, set
      :attr:`Imu.R_base_from_imu` to the correction rotation.
    * Translation offset (lever arm ``r_imu``) does NOT affect ``ang_vel``
      or ``projected_gravity`` — both are frame-invariant under pure
      translation of a rigid body.  So the current observation pipeline
      needs no correction.
    * It DOES affect linear velocity/acceleration if you later add a
      state estimator:

          v_base_origin = v_imu − ω × r_imu
          a_base_origin = a_imu − α × r_imu − ω × (ω × r_imu)

      Use :func:`lever_arm_velocity_correction` for the velocity term.
"""
from __future__ import annotations

import numpy as np


# IMU position relative to base_link origin, in base_link frame (metres).
# From olaf_robstride.urdf imu_joint <origin xyz="-0.095795 0 0.014"/>.
IMU_OFFSET_IN_BASE = np.array([-0.095795, 0.0, 0.014], dtype=np.float32)


def lever_arm_velocity_correction(v_imu_root: np.ndarray,
                                  ang_vel_root: np.ndarray) -> np.ndarray:
    """Transport IMU-site linear velocity to base_link origin.

    All inputs expressed in the root frame.  ``v_base = v_imu − ω × r``.
    """
    return v_imu_root - np.cross(ang_vel_root, IMU_OFFSET_IN_BASE)


class Imu:
    # Set to a 3x3 rotation if your physical mount does not match the URDF.
    # Applied as: root_vec = R_base_from_imu @ sensor_vec.
    R_base_from_imu: np.ndarray = np.eye(3, dtype=np.float32)

    def __init__(self) -> None:
        # TODO: open your sensor (I²C/SPI/UART/USB) and configure output
        # rate + range.  Prefer >= 200 Hz so the policy tick (50 Hz) gets
        # fresh samples with minimal latency.
        pass

    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (ang_vel_root[3], projected_gravity_root[3], yaw_world).

        Implementation notes:

            * Read ω and linear accel from the sensor (sensor frame).
            * Rotate into root: ``ω_root = R_base_from_imu @ ω_sensor``.
            * Estimate orientation (your filter — Madgwick/Mahony/EKF),
              then ``projected_gravity = R_world→root @ [0, 0, -1]``.
            * ``yaw_world`` is the world-frame yaw from the same filter.
        """
        # TODO: replace with real sensor read + orientation integration.
        ang_vel = np.zeros(3, dtype=np.float32)
        proj_g  = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        yaw_w   = 0.0
        return ang_vel, proj_g, yaw_w

    def close(self) -> None:
        pass
