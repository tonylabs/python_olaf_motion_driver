"""Slowly move one or more motors back to their zero positions.

Usage:
    python zero_pos.py <id> [<id> ...] [--model rs-03] [--speed 0.5] [--kp 30.0] [--kd 5.0]

Example:
    python zero_pos.py 1
    python zero_pos.py 1 2 3
    python zero_pos.py 1 2 --model rs-01 --speed 0.3
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from lib.robstride import RobstrideBus, Motor

init(autoreset=True)

KNOWN_MODELS = ["rs-00", "rs-01", "rs-02", "rs-03", "rs-04", "rs-05", "rs-06"]

def normalize_model(value: str) -> str:
    """Normalize model input: 'rs03', 'RS03', 'rs-03' all become 'rs-03'."""
    v = value.strip().lower()
    # Insert dash if missing, e.g. "rs03" -> "rs-03"
    v = re.sub(r"^rs(\d)", r"rs-\1", v)
    if v not in KNOWN_MODELS:
        raise argparse.ArgumentTypeError(
            f"Unknown model '{value}'. Valid models: {', '.join(KNOWN_MODELS)}"
        )
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Slowly move motors to their zero positions")
    parser.add_argument("ids", type=int, nargs="+", help="CAN bus IDs of the motors (1-254)")
    parser.add_argument("--model", type=normalize_model, default="rs-03", help="Motor model, e.g. rs03 or rs-03 (default: rs-03)")
    parser.add_argument("--speed", type=float, default=0.2, help="Max homing speed in rad/s (default: 0.2)")
    parser.add_argument("--kp", type=float, default=30.0, help="Position gain (default: 30.0)")
    parser.add_argument("--kd", type=float, default=5.0, help="Damping gain (default: 5.0)")
    parser.add_argument("--channel", default="can_usb", help="CAN channel (default: can_usb)")
    parser.add_argument("--lock", action="store_true", help="After homing, lock actuators at zero with high torque (Ctrl+C to unlock)")
    args = parser.parse_args()

    dt = 0.01
    motor_names = [f"motor_{mid}" for mid in args.ids]
    motors = {name: Motor(id=mid, model=args.model) for name, mid in zip(motor_names, args.ids)}
    bus = RobstrideBus(args.channel, motors)
    bus.connect()

    # Enable all motors and read starting positions.
    # Ramp from actual reported position toward the nearest multiple of 2π.
    # After a power cycle the multi-turn counter resets, so the reported position
    # can be far from zero. Rounding to the nearest 2π keeps travel within ±π.
    targets = {}
    goals = {}
    for name, mid in zip(motor_names, args.ids):
        bus.enable(name)
        bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0)
        start_pos, _, _, _ = bus.read_operation_frame(name)
        nearest_zero = round(start_pos / (2 * math.pi)) * (2 * math.pi)
        targets[name] = start_pos
        goals[name] = nearest_zero
        distance = start_pos - nearest_zero
        print(
            f"{Fore.LIGHTYELLOW_EX}Motor ID {Fore.WHITE}{Style.BRIGHT}{mid}{Style.RESET_ALL}"
            f"{Fore.LIGHTYELLOW_EX} at {start_pos:+.3f} rad, homing to {nearest_zero:+.3f} rad "
            f"(distance {distance:+.3f} rad){Style.RESET_ALL}"
        )

    print(f"{Fore.CYAN}Homing {len(args.ids)} motor(s) to zero at {args.speed} rad/s...{Style.RESET_ALL}")

    try:
        step = args.speed * dt
        settled = set()
        max_distance = max(abs(targets[n] - goals[n]) for n in motor_names)
        max_iters = int(max_distance / step) + 1000

        for i in range(max_iters):
            for name, mid in zip(motor_names, args.ids):
                if name in settled:
                    continue

                # Ramp target toward goal (nearest multiple of 2π)
                goal = goals[name]
                diff = targets[name] - goal
                if diff > step:
                    targets[name] -= step
                elif diff < -step:
                    targets[name] += step
                else:
                    targets[name] = goal

                bus.write_operation_frame(
                    name, position=targets[name], kp=args.kp, kd=args.kd
                )
                pos, vel, torque, temp = bus.read_operation_frame(name)

                if i % 100 == 0:
                    print(
                        f"  [{mid}] target={targets[name]:+.3f}  pos={pos:+.3f} rad  "
                        f"vel={vel:+.3f} rad/s  torque={torque:+.3f} Nm  temp={temp:.1f}C"
                    )

                if targets[name] == goal and abs(pos - goal) < 0.02 and abs(vel) < 0.3:
                    print(
                        f"  {Fore.GREEN}[{mid}] Reached zero position "
                        f"(pos={pos:+.4f} rad){Style.RESET_ALL}"
                    )
                    settled.add(name)

            # Hold settled motors at goal
            for name in settled:
                bus.write_operation_frame(name, position=goals[name], kp=args.kp, kd=args.kd)
                bus.read_operation_frame(name)

            if len(settled) == len(motor_names):
                # Hold all briefly to stabilize
                for _ in range(100):
                    for name in motor_names:
                        bus.write_operation_frame(name, position=goals[name], kp=args.kp, kd=args.kd)
                        bus.read_operation_frame(name)
                    time.sleep(dt)
                break

            time.sleep(dt)
        else:
            print(f"  {Fore.RED}Timed out before all motors settled at zero.{Style.RESET_ALL}")

        # Lock mode: hold actuators at zero with high stiffness
        if args.lock:
            lock_kp = args.kp * 3
            lock_kd = args.kd * 2
            print(
                f"\n{Fore.RED}{Style.BRIGHT}  ACTUATORS LOCKED at zero position "
                f"(kp={lock_kp}, kd={lock_kd}){Style.RESET_ALL}"
            )
            print(f"{Fore.RED}{Style.BRIGHT}  Press Ctrl+C to unlock and exit{Style.RESET_ALL}\n")
            while True:
                for name in motor_names:
                    bus.write_operation_frame(
                        name, position=goals[name], kp=lock_kp, kd=lock_kd
                    )
                    bus.read_operation_frame(name)
                time.sleep(dt)

    except KeyboardInterrupt:
        print(f"\n{Fore.LIGHTYELLOW_EX}Unlocking actuators...{Style.RESET_ALL}")
    finally:
        bus.disconnect()
        print(f"{Fore.CYAN}Disconnected.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
