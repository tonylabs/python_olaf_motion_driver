"""Set the current position of a DaMiao actuator as the new zero position.

Continuously reads and displays the actuator's current position.
Press 's' to save the current position as the new zero position.

Usage:
    python damiao/zero_pos.py <master_id:can_id> --model 4310
    python damiao/zero_pos.py 0:1 --model 4340P

Example:
    python damiao/zero_pos.py 1:5 --model 4340P
    python damiao/zero_pos.py 2:6 --model 4310
    python damiao/zero_pos.py 3:11 --model 4340P
    python damiao/zero_pos.py 4:12 --model 4310
"""

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from damiao_motor import DaMiaoController

init(autoreset=True)

DEFAULT_MOTOR_TYPE = "4310"

def parse_motor_pair(s: str) -> tuple[int, int]:
    """Parse 'master_id:can_id' string into (feedback_id, motor_id)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected master_id:can_id, got '{s}'")
    return int(parts[0]), int(parts[1])


def key_available() -> str | None:
    """Return the pressed key if one is available, else None (non-blocking)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read position and set zero position for a DaMiao actuator"
    )
    parser.add_argument("motor", type=parse_motor_pair,
                        help="Actuator ID as master_id:can_id (e.g. 0:1)")
    parser.add_argument("--model", default=DEFAULT_MOTOR_TYPE,
                        help=f"Motor model/type (default: {DEFAULT_MOTOR_TYPE})")
    parser.add_argument("--channel", default="can_usb",
                        help="CAN channel (default: can_usb)")
    args = parser.parse_args()

    feedback_id, can_id = args.motor
    label = f"{feedback_id}:{can_id}"

    if sys.platform == "darwin":
        controller = DaMiaoController(
            channel="0046002E594E501820313332", bustype="gs_usb", bitrate=1000000
        )
    else:
        controller = DaMiaoController(channel=args.channel, bustype="socketcan")

    motor = controller.add_motor(
        motor_id=can_id, feedback_id=feedback_id, motor_type=args.model
    )
    motor.enable()

    # Initial position read
    motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
    time.sleep(0.05)
    controller.poll_feedback()

    print(f"\n{Fore.CYAN}Actuator {Fore.WHITE}{Style.BRIGHT}{label}{Style.RESET_ALL}"
          f"{Fore.CYAN} ({args.model}) enabled.{Style.RESET_ALL}")
    print(f"{Fore.LIGHTYELLOW_EX}Press 's' to set current position as new zero. "
          f"Press 'q' or Ctrl+C to quit.{Style.RESET_ALL}\n")

    # Put terminal in raw mode for non-blocking key reads
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            # Send zero-torque command to read feedback
            motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
            controller.poll_feedback()
            pos = motor.state.get("pos", 0.0)
            vel = motor.state.get("vel", 0.0)
            torq = motor.state.get("torq", 0.0)

            sys.stdout.write(
                f"\r  [{label}] pos={pos:+.4f} rad  vel={vel:+.4f} rad/s  "
                f"torque={torq:+.4f} Nm    "
            )
            sys.stdout.flush()

            key = key_available()
            if key == "s":
                sys.stdout.write("\n")
                print(f"{Fore.LIGHTYELLOW_EX}Setting current position ({pos:+.4f} rad) "
                      f"as new zero...{Style.RESET_ALL}")
                motor.set_zero_position()
                time.sleep(0.1)
                controller.poll_feedback()

                # Verify: read position after zeroing
                motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)
                controller.poll_feedback()
                new_pos = motor.state.get("pos", 0.0)
                print(f"{Fore.GREEN}Zero position set. "
                      f"Current position now: {new_pos:+.4f} rad{Style.RESET_ALL}")
                print(f"{Fore.LIGHTYELLOW_EX}Continuing to read position... "
                      f"Press 'q' or Ctrl+C to quit.{Style.RESET_ALL}\n")
            elif key == "q":
                sys.stdout.write("\n")
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        controller.shutdown()
        print(f"{Fore.CYAN}Disconnected.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()