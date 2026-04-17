"""Demo: Robstride Motor Control.

Runs position, velocity, torque, and sine trajectory demos on each motor sequentially.

Usage:
    python robstride/example.py <id> [<id> ...] [--model rs-03] [--channel can_usb]

Example:
    python robstride/example.py 1
    python robstride/example.py 1 2 3
    python robstride/example.py 1 --model rs-01
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from lib.robstride import RobstrideBus, Motor, ParameterType

init(autoreset=True)

KNOWN_MODELS = ["rs-00", "rs-01", "rs-02", "rs-03", "rs-04", "rs-05", "rs-06"]


def normalize_model(value: str) -> str:
    """Normalize model input: 'rs03', 'RS03', 'rs-03' all become 'rs-03'."""
    v = value.strip().lower()
    v = re.sub(r"^rs(\d)", r"rs-\1", v)
    if v not in KNOWN_MODELS:
        raise argparse.ArgumentTypeError(
            f"Unknown model '{value}'. Valid models: {', '.join(KNOWN_MODELS)}"
        )
    return v


def print_status(pos: float, vel: float, torque: float, temp: float) -> None:
    print(
        f"  pos={pos:+.3f} rad  vel={vel:+.3f} rad/s  "
        f"torque={torque:+.3f} Nm  temp={temp:.1f}°C"
    )


def demo_position_control(bus: RobstrideBus, name: str, mid: int) -> None:
    """Move to a target position using PD control."""
    print(f"\n{Fore.CYAN}── [{mid}] Position Control ──{Style.RESET_ALL}")
    target = 1.0
    kp, kd = 30.0, 1.0
    print(f"{Fore.LIGHTYELLOW_EX}Moving to {target} rad (kp={kp}, kd={kd}){Style.RESET_ALL}")

    for _ in range(200):
        bus.write_operation_frame(name, position=target, kp=kp, kd=kd)
        pos, vel, torque, temp = bus.read_operation_frame(name)
        print_status(pos, vel, torque, temp)
        time.sleep(0.01)

    print(f"{Fore.LIGHTYELLOW_EX}Returning to 0 rad{Style.RESET_ALL}")
    for _ in range(200):
        bus.write_operation_frame(name, position=0.0, kp=kp, kd=kd)
        pos, vel, torque, temp = bus.read_operation_frame(name)
        time.sleep(0.01)
    print_status(pos, vel, torque, temp)


def demo_velocity_control(bus: RobstrideBus, name: str, mid: int) -> None:
    """Spin at a constant velocity using kd only."""
    print(f"\n{Fore.CYAN}── [{mid}] Velocity Control ──{Style.RESET_ALL}")
    target_vel = 3.0
    kd = 1.0
    print(f"{Fore.LIGHTYELLOW_EX}Spinning at {target_vel} rad/s (kd={kd}){Style.RESET_ALL}")

    for _ in range(200):
        bus.write_operation_frame(name, position=0.0, kp=0.0, kd=kd, velocity=target_vel)
        pos, vel, torque, temp = bus.read_operation_frame(name)
        print_status(pos, vel, torque, temp)
        time.sleep(0.01)

    print(f"{Fore.LIGHTYELLOW_EX}Stopping{Style.RESET_ALL}")
    for _ in range(100):
        bus.write_operation_frame(name, position=0.0, kp=0.0, kd=kd, velocity=0.0)
        bus.read_operation_frame(name)
        time.sleep(0.01)


def demo_torque_control(bus: RobstrideBus, name: str, mid: int) -> None:
    """Apply a constant feedforward torque."""
    print(f"\n{Fore.CYAN}── [{mid}] Torque Control ──{Style.RESET_ALL}")
    target_torque = 1.0
    print(f"{Fore.LIGHTYELLOW_EX}Applying {target_torque} Nm feedforward torque{Style.RESET_ALL}")

    for _ in range(200):
        bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0, torque=target_torque)
        pos, vel, torque, temp = bus.read_operation_frame(name)
        print_status(pos, vel, torque, temp)
        time.sleep(0.01)

    for _ in range(50):
        bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0, torque=0.0)
        bus.read_operation_frame(name)
        time.sleep(0.01)


def demo_sine_trajectory(bus: RobstrideBus, name: str, mid: int) -> None:
    """Track a smooth sinusoidal position trajectory."""
    print(f"\n{Fore.CYAN}── [{mid}] Sine Trajectory ──{Style.RESET_ALL}")
    freq = 0.25
    amplitude = 1.0
    kp, kd = 30.0, 2.0
    duration = 4.0
    dt = 0.01
    steps = int(duration / dt)
    print(f"{Fore.LIGHTYELLOW_EX}Sine wave: amplitude={amplitude} rad, freq={freq} Hz, duration={duration}s{Style.RESET_ALL}")

    t0 = time.time()
    for i in range(steps):
        t = i * dt
        target = amplitude * math.sin(2 * math.pi * freq * t)
        target_vel = amplitude * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t)

        bus.write_operation_frame(name, position=target, kp=kp, kd=kd, velocity=target_vel)
        pos, vel, torque, temp = bus.read_operation_frame(name)

        if i % 50 == 0:
            print(f"  t={t:.2f}s target={target:+.3f} rad", end="  ")
            print_status(pos, vel, torque, temp)

        elapsed = time.time() - t0
        sleep_time = (i + 1) * dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def demo_read_parameters(bus: RobstrideBus, name: str, mid: int) -> None:
    """Read various motor parameters."""
    print(f"\n{Fore.CYAN}── [{mid}] Motor Parameters ──{Style.RESET_ALL}")
    params = [
        ("Position", ParameterType.MEASURED_POSITION),
        ("Velocity", ParameterType.MEASURED_VELOCITY),
        ("Torque", ParameterType.MEASURED_TORQUE),
        ("Bus Voltage", ParameterType.VBUS),
        ("Torque Limit", ParameterType.TORQUE_LIMIT),
        ("Velocity Limit", ParameterType.VELOCITY_LIMIT),
    ]
    for param_name, param in params:
        value = bus.read(name, param)
        print(f"  {Fore.LIGHTYELLOW_EX}{param_name}: {Fore.WHITE}{Style.BRIGHT}{value}{Style.RESET_ALL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run motor control demos on Robstride actuators")
    parser.add_argument("ids", type=int, nargs="+", help="CAN bus IDs of the motors (1-254)")
    parser.add_argument("--model", type=normalize_model, default="rs-03", help="Motor model (default: rs-03)")
    parser.add_argument("--channel", default="can_usb", help="CAN channel (default: can_usb)")
    args = parser.parse_args()

    motor_names = [f"motor_{mid}" for mid in args.ids]
    motors = {name: Motor(id=mid, model=args.model) for name, mid in zip(motor_names, args.ids)}
    bus = RobstrideBus(args.channel, motors)

    try:
        bus.connect()
        print(f"{Fore.GREEN}Connected to CAN bus '{args.channel}'{Style.RESET_ALL}")

        for name, mid in zip(motor_names, args.ids):
            bus.enable(name)
            print(f"{Fore.GREEN}Motor {mid} enabled{Style.RESET_ALL}")

            demo_read_parameters(bus, name, mid)
            demo_position_control(bus, name, mid)
            demo_velocity_control(bus, name, mid)
            demo_torque_control(bus, name, mid)
            demo_sine_trajectory(bus, name, mid)

    except KeyboardInterrupt:
        print(f"\n{Fore.LIGHTYELLOW_EX}Interrupted by user{Style.RESET_ALL}")
    finally:
        bus.disconnect()
        print(f"{Fore.CYAN}Disconnected{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
