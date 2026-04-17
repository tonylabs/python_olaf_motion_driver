"""Demo: DaMiao 6006 Motor Control via damiao-motor.

Setup (Raspberry Pi + CANable 2.0 with candlelight firmware):
    sudo ip link set can_usb type can bitrate 1000000
    sudo ip link set can_usb up
"""

import math
import sys
import time

from damiao_motor import DaMiaoController

# ── Configuration ──────────────────────────────────────────────
MOTOR_ID = 0x01
FEEDBACK_ID = 0x00
MOTOR_TYPE = "6006"
# Available motor types: 3507, 4310, 4340, 6006, 8006, 8009, 10010L,
# 10010, H3510, G6215, H6220, JH11, 6248P

if sys.platform == "darwin":
    controller = DaMiaoController(
        channel="0046002E594E501820313332", bustype="gs_usb", bitrate=1000000
    )
else:
    controller = DaMiaoController(channel="can_usb", bustype="socketcan")

motor = controller.add_motor(
    motor_id=MOTOR_ID, feedback_id=FEEDBACK_ID, motor_type=MOTOR_TYPE
)


def read_feedback() -> tuple[float, float, float]:
    """Poll feedback and return (pos, vel, torque)."""
    controller.poll_feedback()
    pos = motor.state.get("pos", 0.0)
    vel = motor.state.get("vel", 0.0)
    torq = motor.state.get("torq", 0.0)
    return pos, vel, torq


def print_status(pos: float, vel: float, torque: float) -> None:
    print(
        f"  pos={pos:+.3f} rad  vel={vel:+.3f} rad/s  "
        f"torque={torque:+.3f} Nm"
    )


def demo_position_control() -> None:
    """Move to a target position using PD control (stiffness + damping)."""
    print("\n── Position Control ──")
    target = 1.0  # radians
    kp, kd = 30.0, 1.0
    print(f"Moving to {target} rad (kp={kp}, kd={kd})")

    for _ in range(200):
        motor.send_cmd_mit(
            target_position=target,
            target_velocity=0.0,
            stiffness=kp,
            damping=kd,
            feedforward_torque=0.0,
        )
        pos, vel, torq = read_feedback()
        print_status(pos, vel, torq)
        time.sleep(0.01)

    # Return to zero
    print("Returning to 0 rad")
    for _ in range(200):
        motor.send_cmd_mit(
            target_position=0.0,
            target_velocity=0.0,
            stiffness=kp,
            damping=kd,
            feedforward_torque=0.0,
        )
        pos, vel, torq = read_feedback()
        time.sleep(0.01)
    print_status(pos, vel, torq)


def demo_velocity_control() -> None:
    """Spin at a constant velocity using damping only."""
    print("\n── Velocity Control ──")
    target_vel = 3.0  # rad/s
    kd = 1.0
    print(f"Spinning at {target_vel} rad/s (kd={kd})")

    for _ in range(200):
        motor.send_cmd_mit(
            target_position=0.0,
            target_velocity=target_vel,
            stiffness=0.0,
            damping=kd,
            feedforward_torque=0.0,
        )
        pos, vel, torq = read_feedback()
        print_status(pos, vel, torq)
        time.sleep(0.01)

    # Stop
    print("Stopping")
    for _ in range(100):
        motor.send_cmd_mit(
            target_position=0.0,
            target_velocity=0.0,
            stiffness=0.0,
            damping=kd,
            feedforward_torque=0.0,
        )
        read_feedback()
        time.sleep(0.01)


def demo_torque_control() -> None:
    """Apply a constant feedforward torque."""
    print("\n── Torque Control ──")
    target_torque = 1.0  # Nm
    print(f"Applying {target_torque} Nm feedforward torque")

    for _ in range(200):
        motor.send_cmd_mit(
            target_position=0.0,
            target_velocity=0.0,
            stiffness=0.0,
            damping=0.0,
            feedforward_torque=target_torque,
        )
        pos, vel, torq = read_feedback()
        print_status(pos, vel, torq)
        time.sleep(0.01)

    # Release
    for _ in range(50):
        motor.send_cmd_mit(
            target_position=0.0,
            target_velocity=0.0,
            stiffness=0.0,
            damping=0.0,
            feedforward_torque=0.0,
        )
        read_feedback()
        time.sleep(0.01)


def demo_sine_trajectory() -> None:
    """Track a smooth sinusoidal position trajectory."""
    print("\n── Sine Trajectory ──")
    freq = 0.25  # Hz
    amplitude = 1.0  # radians
    kp, kd = 30.0, 2.0
    duration = 4.0  # seconds (1 full cycle)
    dt = 0.01
    steps = int(duration / dt)
    print(f"Sine wave: amplitude={amplitude} rad, freq={freq} Hz, duration={duration}s")

    t0 = time.time()
    for i in range(steps):
        t = i * dt
        target = amplitude * math.sin(2 * math.pi * freq * t)
        target_vel = amplitude * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t)

        motor.send_cmd_mit(
            target_position=target,
            target_velocity=target_vel,
            stiffness=kp,
            damping=kd,
            feedforward_torque=0.0,
        )
        pos, vel, torq = read_feedback()

        if i % 50 == 0:
            print(f"  t={t:.2f}s target={target:+.3f} rad", end="  ")
            print_status(pos, vel, torq)

        elapsed = time.time() - t0
        sleep_time = (i + 1) * dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    try:
        print(f"Motor {MOTOR_TYPE} (ID={MOTOR_ID}) ready")

        demo_position_control()
        demo_velocity_control()
        demo_torque_control()
        demo_sine_trajectory()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        controller.shutdown()
        print("Disconnected")


if __name__ == "__main__":
    main()
