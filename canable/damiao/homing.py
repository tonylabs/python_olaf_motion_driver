"""Slowly move one or more DaMiao actuators back to their zero positions.

Usage:
    python damiao/zero_pos.py <master_id:can_id> [<master_id:can_id> ...] [--speed 0.2]

Example:
    python damiao/zero_pos.py 0:1
    python damiao/zero_pos.py 0:1 0:2 0:3
    python damiao/zero_pos.py 0:1 1:2 --speed 0.1
"""

import argparse
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from damiao_motor import DaMiaoController
from damiao_motor.core.motor import MOTOR_TYPE_PRESETS

init(autoreset=True)

ACTUATOR_TYPES = sorted(MOTOR_TYPE_PRESETS.keys())


def parse_motor_pair(s: str) -> tuple[int, int]:
    """Parse 'master_id:can_id' string into (feedback_id, motor_id)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected master_id:can_id, got '{s}'")
    return int(parts[0]), int(parts[1])


def select_actuator_type():
    """Let user pick an actuator type from the available presets."""
    print(f"\n{Fore.LIGHTYELLOW_EX}Available actuator types:{Style.RESET_ALL}")
    for idx, atype in enumerate(ACTUATOR_TYPES, 1):
        print(f"  {Fore.WHITE}{Style.BRIGHT}{idx:>2}{Style.RESET_ALL}) {atype}")

    while True:
        raw = input(f"\nSelect actuator type [1-{len(ACTUATOR_TYPES)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(ACTUATOR_TYPES):
                selected = ACTUATOR_TYPES[choice - 1]
                print(f"{Fore.GREEN}Selected: {selected}{Style.RESET_ALL}")
                return selected
            print(f"{Fore.RED}Please enter a number between 1 and {len(ACTUATOR_TYPES)}.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slowly move DaMiao actuators to their zero positions")
    parser.add_argument("motors", type=parse_motor_pair, nargs="+",
                        help="Actuator IDs as master_id:can_id (e.g. 0:1)")
    parser.add_argument("--speed", type=float, default=0.2, help="Max homing speed in rad/s (default: 0.2)")
    parser.add_argument("--kp", type=float, default=30.0, help="Stiffness gain (default: 30.0)")
    parser.add_argument("--kd", type=float, default=1.0, help="Damping gain (default: 1.0, max 5.0)")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Position tolerance in rad (default: 0.15)")
    parser.add_argument("--channel", default="can_usb", help="CAN channel (default: can_usb)")
    args = parser.parse_args()

    # Select actuator type interactively
    actuator_type = select_actuator_type()

    dt = 0.01

    if sys.platform == "darwin":
        controller = DaMiaoController(
            channel="0046002E594E501820313332", bustype="gs_usb", bitrate=1000000
        )
    else:
        controller = DaMiaoController(channel=args.channel, bustype="socketcan")

    # Add and enable all actuators, read starting positions.
    # Ramp from actual reported position toward the nearest multiple of 2π.
    # After a power cycle the multi-turn counter resets, so the reported position
    # can be far from zero. Rounding to the nearest 2π keeps travel within ±π.
    motor_map = {}
    targets = {}
    goals = {}
    motor_labels = []
    for feedback_id, can_id in args.motors:
        motor = controller.add_motor(motor_id=can_id, feedback_id=feedback_id, motor_type=actuator_type)
        motor.enable()
        # Read current position with zero gains (no force)
        motor.send_cmd_mit(
            target_position=0.0, target_velocity=0.0,
            stiffness=0.0, damping=0.0, feedforward_torque=0.0,
        )
        time.sleep(0.05)
        controller.poll_feedback()
        start_pos = motor.state.get("pos", 0.0)
        nearest_zero = round(start_pos / (2 * math.pi)) * (2 * math.pi)
        label = f"{feedback_id}:{can_id}"
        motor_map[label] = motor
        targets[label] = start_pos
        goals[label] = nearest_zero
        distance = start_pos - nearest_zero
        motor_labels.append(label)
        print(
            f"{Fore.LIGHTYELLOW_EX}Actuator {Fore.WHITE}{Style.BRIGHT}{label}{Style.RESET_ALL}"
            f"{Fore.LIGHTYELLOW_EX} ({actuator_type}) at {start_pos:+.3f} rad, "
            f"homing to {nearest_zero:+.3f} rad (distance {distance:+.3f} rad){Style.RESET_ALL}"
        )

    print(f"{Fore.CYAN}Homing {len(motor_labels)} actuator(s) to zero at {args.speed} rad/s...{Style.RESET_ALL}")

    try:
        step = args.speed * dt
        settled = set()
        max_distance = max(abs(targets[n] - goals[n]) for n in motor_labels)
        max_iters = int(max_distance / step) + 1000

        for i in range(max_iters):
            for label in motor_labels:
                if label in settled:
                    continue

                motor = motor_map[label]
                goal = goals[label]

                # Ramp target toward goal (nearest multiple of 2π)
                diff = targets[label] - goal
                if diff > step:
                    targets[label] -= step
                elif diff < -step:
                    targets[label] += step
                else:
                    targets[label] = goal

                motor.send_cmd_mit(
                    target_position=targets[label],
                    target_velocity=0.0,
                    stiffness=args.kp,
                    damping=args.kd,
                    feedforward_torque=0.0,
                )
                controller.poll_feedback()
                pos = motor.state.get("pos", 0.0)
                vel = motor.state.get("vel", 0.0)
                torq = motor.state.get("torq", 0.0)

                if i % 100 == 0:
                    print(
                        f"  [{label}] target={targets[label]:+.3f}  pos={pos:+.3f} rad  "
                        f"vel={vel:+.3f} rad/s  torque={torq:+.3f} Nm"
                    )

                if targets[label] == goal and abs(pos - goal) < args.tolerance and abs(vel) < 0.3:
                    print(
                        f"  {Fore.GREEN}[{label}] Reached zero position "
                        f"(pos={pos:+.4f} rad){Style.RESET_ALL}"
                    )
                    settled.add(label)

            # Hold settled actuators at goal
            for label in settled:
                motor_map[label].send_cmd_mit(
                    target_position=goals[label], target_velocity=0.0,
                    stiffness=args.kp, damping=args.kd, feedforward_torque=0.0,
                )
                controller.poll_feedback()

            if len(settled) == len(motor_labels):
                break

            time.sleep(dt)
        else:
            print(f"  {Fore.RED}Timed out before all actuators settled at zero.{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.LIGHTYELLOW_EX}Interrupted.{Style.RESET_ALL}")
    finally:
        controller.shutdown()
        print(f"{Fore.CYAN}Disconnected.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
