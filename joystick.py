"""Xbox One S controller → robot velocity command.

The left stick drives the robot:
    left-stick up    →  +vx (forward)
    left-stick down  →  -vx (backward)
    left-stick left  →  +vy (left)
    left-stick right →  -vy (right)

Convention follows ROS REP-103: +x forward, +y left. Stick axes are inverted
from SDL's raw values (SDL: up = -1) to give this convention directly.

Use from `run.py` like:

    joy = Joystick(max_lin_vel=0.8)
    joy.start()
    ...
    self._velocity_cmd = joy.velocity_cmd    # in policy tick
    ...
    joy.stop()

Run standalone to verify mapping:

    python joystick.py
"""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

_AXIS_LEFT_X = 0
_AXIS_LEFT_Y = 1


class Joystick:
    """Background-polled Xbox controller. Exposes a 4-vector
    `velocity_cmd = [vx, vy, wz, heading]` matching the policy's obs.

    Only vx / vy are driven by the left stick; wz and heading stay zero.
    """

    def __init__(self,
                 device: int = 0,
                 max_lin_vel: float = 1.0,
                 deadzone: float = 0.08,
                 poll_hz: float = 100.0):
        self._device = device
        self._max_v = float(max_lin_vel)
        self._deadzone = float(deadzone)
        self._dt = 1.0 / float(poll_hz)

        self._cmd = np.zeros(4, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._js: pygame.joystick.JoystickType | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self._device:
            raise RuntimeError(
                f"Joystick {self._device} not found "
                f"(pygame sees {pygame.joystick.get_count()} devices)"
            )
        self._js = pygame.joystick.Joystick(self._device)
        self._js.init()

        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="joystick-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._js is not None:
            self._js.quit()
        pygame.quit()

    # -- public state -------------------------------------------------------
    @property
    def velocity_cmd(self) -> np.ndarray:
        """(4,) float32 snapshot of [vx, vy, wz, heading]. Safe to read."""
        with self._lock:
            return self._cmd.copy()

    # -- internals ----------------------------------------------------------
    def _apply_deadzone(self, v: float) -> float:
        if abs(v) < self._deadzone:
            return 0.0
        # Rescale outside deadzone to span [0, 1] so the stick feels full-range
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - self._deadzone) / (1.0 - self._deadzone)

    def _loop(self) -> None:
        assert self._js is not None
        while not self._stop.is_set():
            pygame.event.pump()   # required for axis values to refresh
            lx = self._apply_deadzone(self._js.get_axis(_AXIS_LEFT_X))
            ly = self._apply_deadzone(self._js.get_axis(_AXIS_LEFT_Y))
            # SDL: stick up = -1, stick right = +1. Flip both so the robot
            # convention (forward = +vx, left = +vy) matches the user.
            vx = -ly * self._max_v
            vy = -lx * self._max_v
            with self._lock:
                self._cmd[0] = vx
                self._cmd[1] = vy
                # wz and heading stay zero — not driven by left stick
            time.sleep(self._dt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    args = parser.parse_args()

    joy = Joystick(device=args.device,
                   max_lin_vel=args.max_vel,
                   deadzone=args.deadzone)
    joy.start()
    print(f"Joystick started. max_vel={args.max_vel} m/s, "
          f"deadzone={args.deadzone}. Ctrl+C to stop.")
    try:
        while True:
            cmd = joy.velocity_cmd
            print(f"\rvx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  "
                  f"wz={cmd[2]:+.2f}  hdg={cmd[3]:+.2f}",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        joy.stop()


if __name__ == "__main__":
    main()
