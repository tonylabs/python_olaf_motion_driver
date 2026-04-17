"""Set the current position of a Robstride motor as the new zero position.

Continuously reads and displays the motor's current position.
Press 's' to save the current position as the new zero position.

Usage:
    python robstride/zero_pos.py <motor_id> --model rs-03
    python robstride/zero_pos.py 1 --model rs-01
"""

import argparse
import re
import select
import sys
import termios
import time
import tty
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from lib.robstride import RobstrideBus, Motor
from lib.robstride.protocol import CommunicationType

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


def key_available() -> str | None:
    """Return the pressed key if one is available, else None (non-blocking)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read position and set zero position for a Robstride motor"
    )
    parser.add_argument("motor_id", type=int,
                        help="CAN bus ID of the motor (1-254)")
    parser.add_argument("--model", type=normalize_model, default="rs-03",
                        help="Motor model, e.g. rs03 or rs-03 (default: rs-03)")
    parser.add_argument("--channel", default="can_usb",
                        help="CAN channel (default: can_usb)")
    args = parser.parse_args()

    mid = args.motor_id
    name = f"motor_{mid}"
    motors = {name: Motor(id=mid, model=args.model)}
    bus = RobstrideBus(args.channel, motors)
    bus.connect()

    bus.enable(name)

    # Initial position read
    bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0)
    pos, vel, torque, temp = bus.read_operation_frame(name)

    print(f"\n{Fore.CYAN}Motor {Fore.WHITE}{Style.BRIGHT}{mid}{Style.RESET_ALL}"
          f"{Fore.CYAN} ({args.model}) enabled.{Style.RESET_ALL}")
    print(f"{Fore.LIGHTYELLOW_EX}Press 's' to set current position as new zero. "
          f"Press 'q' or Ctrl+C to quit.{Style.RESET_ALL}\n")

    # Put terminal in raw mode for non-blocking key reads
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            # Send zero-torque command to read feedback
            bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0)
            pos, vel, torque, temp = bus.read_operation_frame(name)

            sys.stdout.write(
                f"\r  [{mid}] pos={pos:+.4f} rad  vel={vel:+.4f} rad/s  "
                f"torque={torque:+.4f} Nm  temp={temp:.1f}C    "
            )
            sys.stdout.flush()

            key = key_available()
            if key == "s":
                sys.stdout.write("\n")
                print(f"{Fore.LIGHTYELLOW_EX}Setting current position ({pos:+.4f} rad) "
                      f"as new zero...{Style.RESET_ALL}")
                bus.transmit(CommunicationType.SET_ZERO_POSITION, bus.host_id, mid,
                             data=b"\x01\x00\x00\x00\x00\x00\x00\x00")
                bus.receive()  # consume the response frame
                time.sleep(0.1)

                # Re-enable motor after zero position set
                bus.enable(name)

                # Verify: read position after zeroing
                bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0)
                new_pos, _, _, _ = bus.read_operation_frame(name)
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
        bus.disconnect()
        print(f"{Fore.CYAN}Disconnected.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
