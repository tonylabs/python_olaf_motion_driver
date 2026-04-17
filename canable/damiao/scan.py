"""Scan the CAN bus for DaMiao actuators.

Probes for all connected actuators regardless of type, displays a clear
ID summary (CAN_ID and MASTER_ID), then shows all register parameters.

The actuator type only affects encoding of the zero probe command — since
all zeros encode identically across types, detection works for any model.

Usage:
    python damiao/scan.py [--channel CAN0] [--start 1] [--end 32] [--model 4310]
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colorama import Fore, Style, init
from damiao_motor.cli.display import scan_motors
from damiao_motor import DaMiaoController

init(autoreset=True)

# Available motor types: 3507, 4310, 4340, 6006, 8006, 8009, 10010L,
# 10010, H3510, G6215, H6220, JH11, 6248P
DEFAULT_MOTOR_TYPE = "4310"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan the CAN bus for DaMiao actuators.")
    parser.add_argument("--channel", default="can_usb", help="CAN channel (default: can_usb)")
    parser.add_argument("--start", type=int, default=1, help="Start motor ID (default: 1)")
    parser.add_argument("--end", type=int, default=32, help="End motor ID (default: 32)")
    parser.add_argument("--model", default=DEFAULT_MOTOR_TYPE,
                        help=f"Motor model/type (default: {DEFAULT_MOTOR_TYPE})")
    args = parser.parse_args()

    channel = args.channel
    start = args.start
    end = args.end
    motor_type = args.model

    motor_ids = list(range(start, end))

    print(f"{Fore.CYAN}Scanning {channel} for DaMiao actuators (IDs 0x{start:02X}-0x{end-1:02X}, model={motor_type})...{Style.RESET_ALL}")

    # Library scan — detects all actuator types, shows full register table
    responded = scan_motors(
        channel=channel, motor_ids=motor_ids, duration_s=3.0,
        motor_type=motor_type,
    )

    if not responded:
        print(f"\n{Fore.RED}No actuators found.{Style.RESET_ALL}")
        sys.exit(1)

    # Read CAN_ID and MASTER_ID from each actuator for a clear summary
    controller = DaMiaoController(channel=channel, bustype="socketcan")
    controller.flush_bus()

    motors = {}
    for mid in sorted(responded):
        try:
            motor = controller.add_motor(
                motor_id=mid, feedback_id=0x00, motor_type=motor_type
            )
            motors[mid] = motor
        except ValueError:
            pass

    # Send zero commands and collect feedback to get MASTER_ID from frames
    for motor in motors.values():
        try:
            motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
    time.sleep(0.1)
    controller.poll_feedback()

    # Try reading CAN_ID (reg 8) and MASTER_ID (reg 7) via registers.
    # Older firmware (e.g. DM-J4310-2EC V1.2) doesn't support the register
    # read protocol — fall back to scan probe ID and feedback frame data.
    print(f"\n{Fore.CYAN}Reading CAN_ID and MASTER_ID registers...{Style.RESET_ALL}")
    for mid in sorted(motors.keys()):
        motor = motors[mid]
        try:
            motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        controller.poll_feedback()
        motor.request_register_reading(8)
        time.sleep(0.05)
        controller.poll_feedback()
        motor.request_register_reading(7)
        time.sleep(0.05)
        controller.poll_feedback()

    # Extra polling pass
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        for motor in motors.values():
            try:
                motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass
        controller.poll_feedback()
        time.sleep(0.05)

    # Print ID summary
    print(f"\n{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}  Actuator ID Summary{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}\n")
    print(f"  {Fore.CYAN}{'Actuator':<12}{'CAN_ID':<25}{'MASTER_ID (feedback)':<20}{Style.RESET_ALL}")
    print(f"  {'─' * 55}")

    for mid in sorted(motors.keys()):
        motor = motors[mid]
        reg_ok = True

        # Try register read for CAN_ID
        try:
            can_id = int(motor.get_register(8, timeout=0.1))
            can_str = f"0x{can_id:02X} ({can_id})"
        except Exception:
            reg_ok = False
            # Fallback: the probe ID is the CAN_ID we used to reach it
            can_str = f"0x{mid:02X} ({mid})"

        # Try register read for MASTER_ID
        try:
            master_id = int(motor.get_register(7, timeout=0.1))
            master_str = f"0x{master_id:02X} ({master_id})"
        except Exception:
            # Fallback: extract from feedback frame (data[0] & 0x0F)
            fb_id = motor.state.get("can_id") if motor.state else None
            if fb_id is not None:
                master_str = f"0x{fb_id:02X} ({fb_id})"
            else:
                master_str = "N/A"

        note = ""
        if not reg_ok:
            note = f"  {Fore.LIGHTYELLOW_EX}(old firmware, no register read){Style.RESET_ALL}"

        print(
            f"  {Fore.WHITE}{Style.BRIGHT}0x{mid:02X} ({mid}){Style.RESET_ALL}"
            f"   {Fore.LIGHTYELLOW_EX}{can_str:<25}{Style.RESET_ALL}"
            f"{Fore.LIGHTYELLOW_EX}{master_str:<20}{Style.RESET_ALL}"
            f"{note}"
        )

    print()

    try:
        controller.disable_all()
        controller.bus.shutdown()
    except Exception:
        pass