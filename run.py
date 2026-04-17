"""Main runtime entry — 50 Hz policy tick, 600 Hz motor tick.

Threads:
    * MAIN              — policy inference @ 50 Hz, writes q_target
    * motor thread      — upsamples q_target to 600 Hz, LPF, sends CAN

The two share ``_latest_target`` and ``_last_policy_t`` under a lock.  The
motor thread holds the hot path; keep Python objects off it.

On a non-RT Linux the 600 Hz tick will jitter visibly (±2–5 ms).  Use a
PREEMPT_RT kernel and ``chrt -f 90`` for serious work.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

# Suppress ONNX Runtime GPU discovery warning on headless / GPU-less boards
os.environ.setdefault("ORT_DISABLE_GPU_DEVICE_ENUMERATION", "1")

import numpy as np

from can_bus import CanBus
from config import (
    DEFAULT_JOINT_POS,
    JOINT_ORDER,
    KP_RAMP_S,
    LPF_CUTOFF_HZ,
    MOTOR_DT,
    N_JOINTS,
    POLICY_DT,
    POLICY_WATCHDOG_S,
)
from imu import Imu
from motors import MotorBus
from observation import ObservationBuilder, OBS_DIM
from policy import DeployedPolicy

log = logging.getLogger("olaf.run")


class FirstOrderLowPass:
    """Per-joint LPF.  α = dt / (RC + dt), RC = 1 / (2π f_c)."""

    def __init__(self, cutoff_hz: float, dt: float, n: int):
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        self.alpha = dt / (rc + dt)
        self.y = np.zeros(n, dtype=np.float32)

    def reset(self, x: np.ndarray) -> None:
        self.y = x.astype(np.float32).copy()

    def step(self, x: np.ndarray) -> np.ndarray:
        self.y += self.alpha * (x - self.y)
        return self.y


class Runtime:
    def __init__(self, policy_dir: Path, can_channel: str = "can_usb"):
        self._bus = CanBus(channel=can_channel, bitrate=1_000_000)
        self._motors = MotorBus(self._bus, dm_channel=can_channel)
        self._imu = Imu()
        self._policy = DeployedPolicy(policy_dir)
        self._obs_builder = ObservationBuilder()

        # Shared state between policy tick and motor tick
        self._lock = threading.Lock()
        self._target_prev = DEFAULT_JOINT_POS.copy()
        self._target_curr = DEFAULT_JOINT_POS.copy()
        self._target_t = time.monotonic()
        self._kp_scale = 0.0
        self._stop = threading.Event()

        # Command — exposed for a joystick hook to mutate.
        self._velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._lpf = FirstOrderLowPass(LPF_CUTOFF_HZ, MOTOR_DT, N_JOINTS)
        self._lpf.reset(DEFAULT_JOINT_POS)

    # -- motor thread (600 Hz) ---------------------------------------------
    def _motor_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                target_prev = self._target_prev
                target_curr = self._target_curr
                t0 = self._target_t
                kp_scale = self._kp_scale
                stale_for = now - t0

            # Watchdog — drop to damping if policy is missing ticks
            if stale_for > POLICY_WATCHDOG_S * 2:
                self._motors.damp_all()
            else:
                # First-order hold from prev → curr over POLICY_DT
                alpha = min(max(stale_for / POLICY_DT, 0.0), 1.0)
                q_interp = (1.0 - alpha) * target_prev + alpha * target_curr
                q_filt = self._lpf.step(q_interp)
                self._motors.command(q_filt, np.zeros(N_JOINTS, dtype=np.float32),
                                     kp_scale=kp_scale)

            # Non-blocking drain of feedback
            self._motors.pump_feedback(budget_s=0.0005)

            next_t += MOTOR_DT
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # we're behind; resync

    # -- policy tick (50 Hz) -----------------------------------------------
    def _policy_tick(self) -> None:
        ang_vel, proj_g, _yaw_w = self._imu.read()
        q  = self._motors.joint_pos
        qd = self._motors.joint_vel

        obs = self._obs_builder.build(
            velocity_cmd=self._velocity_cmd,
            base_ang_vel_root=ang_vel,
            projected_gravity=proj_g,
            joint_pos=q,
            joint_vel=qd,
        )
        assert obs.shape == (OBS_DIM,)

        action = self._policy(obs)                    # raw action
        q_target = action + DEFAULT_JOINT_POS         # match ActionsCfg

        self._obs_builder.push_action(action)
        self._obs_builder.step_phase()

        with self._lock:
            self._target_prev = self._target_curr
            self._target_curr = q_target.astype(np.float32)
            self._target_t = time.monotonic()

    # -- lifecycle ----------------------------------------------------------
    def run(self) -> None:
        self._motors.enable_all()
        self._motors.command(DEFAULT_JOINT_POS, np.zeros(N_JOINTS, dtype=np.float32), kp_scale=0.0)

        mot = threading.Thread(target=self._motor_loop, daemon=True)
        mot.start()

        signal.signal(signal.SIGINT, lambda *_: self._stop.set())

        t_boot = time.monotonic()
        next_t = time.monotonic()
        try:
            while not self._stop.is_set():
                # Soft-start: ramp kp from 0 → 1 over KP_RAMP_S
                ramp = min((time.monotonic() - t_boot) / KP_RAMP_S, 1.0)
                with self._lock:
                    self._kp_scale = ramp

                t0 = time.monotonic()
                self._policy_tick()
                tick_s = time.monotonic() - t0
                if tick_s > POLICY_WATCHDOG_S:
                    log.warning("policy tick %.1f ms > watchdog", tick_s * 1e3)

                next_t += POLICY_DT
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()
        finally:
            self._stop.set()
            mot.join(timeout=0.5)
            self._motors.disable_all()
            self._bus.close()
            self._imu.close()


DEPLOY_DIR = Path(__file__).resolve().parent / "scripts" / "deploy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, default=DEPLOY_DIR,
                        help="Directory with policy.onnx + preprocessor.json "
                             "(default: scripts/deploy/)")
    parser.add_argument("--can", default="can_usb")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    assert len(JOINT_ORDER) == N_JOINTS
    rt = Runtime(args.policy_dir, can_channel=args.can)
    rt.run()


if __name__ == "__main__":
    main()
