"""Change the CAN bus ID of a Robstride motor.

Scans for connected motors, lets the user pick one and assign a new ID,
then scans again to confirm the change took effect.

Usage:
    python robstride/chg_id.py [channel]

Example:
    python robstride/chg_id.py
    python robstride/chg_id.py can1
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import can
from colorama import Fore, Style, init
from lib.robstride import RobstrideBus, Motor

init(autoreset=True)


def _patched_transmit(original_transmit):
    def wrapper(self, *args, **kwargs):
        for attempt in range(10):
            try:
                return original_transmit(self, *args, **kwargs)
            except can.CanOperationError:
                time.sleep(0.05 * (attempt + 1))
        print(f"{Fore.LIGHTYELLOW_EX}WARNING: Could not send after retries (buffer full){Style.RESET_ALL}")
    return wrapper


# Patch transmit to handle buffer full
RobstrideBus.transmit = _patched_transmit(RobstrideBus.transmit)


def scan_motors(channel, start=1, end=128):
    """Scan and return dict of {id: response}."""
    print(f"\n{Fore.CYAN}Scanning {channel} for motor IDs {start}-{end - 1}...{Style.RESET_ALL}")
    motors = RobstrideBus.scan_channel(channel, start, end)
    if motors:
        print(f"{Fore.GREEN}Found {len(motors)} motor(s): {list(motors.keys())}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No motors found.{Style.RESET_ALL}")
    return motors


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else "can_usb"

    # Step 1: Scan for connected motors
    found = scan_motors(channel)
    if not found:
        return

    # Step 2: Ask user which motor to change
    found_ids = sorted(found.keys())
    print(f"\n{Fore.LIGHTYELLOW_EX}Available motors:{Style.RESET_ALL}")
    for idx, mid in enumerate(found_ids, 1):
        print(f"  {Fore.WHITE}{Style.BRIGHT}{idx}{Style.RESET_ALL}) Motor ID {Fore.WHITE}{Style.BRIGHT}{mid}{Style.RESET_ALL}")

    while True:
        raw = input(f"\nSelect motor to change [1-{len(found_ids)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(found_ids):
                old_id = found_ids[choice - 1]
                break
            print(f"{Fore.RED}Please enter a number between 1 and {len(found_ids)}.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

    # Step 3: Ask for the new ID
    while True:
        raw = input(f"Enter the new ID for motor {old_id} (1-128): ").strip()
        try:
            new_id = int(raw)
            if not (1 <= new_id <= 128):
                print(f"{Fore.RED}ID must be between 1 and 128.{Style.RESET_ALL}")
                continue
            if new_id == old_id:
                print(f"{Fore.RED}New ID is the same as the current ID ({old_id}). Try again.{Style.RESET_ALL}")
                continue
            if new_id in found_ids:
                print(f"{Fore.RED}ID {new_id} is already in use by another motor. Try again.{Style.RESET_ALL}")
                continue
            break
        except ValueError:
            print(f"{Fore.RED}Please enter a valid integer.{Style.RESET_ALL}")

    # Step 4: Change the ID
    print(f"\n{Fore.CYAN}Changing motor ID from {old_id} to {new_id}...{Style.RESET_ALL}")
    motor_name = "target"
    motors = {motor_name: Motor(id=old_id, model="rs-03")}
    bus = RobstrideBus(channel, motors)
    bus.connect(handshake=False)

    result = bus.write_id(motor_name, new_id)

    if result is None:
        bus.disconnect(disable_torque=False)
        print(f"{Fore.RED}ERROR: No response from motor after ID change command.{Style.RESET_ALL}")
        return

    print(f"{Fore.LIGHTYELLOW_EX}Motor responded with new ID: {Fore.WHITE}{Style.BRIGHT}{result[0]}{Style.RESET_ALL}")

    # Flush any stale frames before closing
    while bus.channel_handler.recv(timeout=0.1):
        pass
    bus.disconnect(disable_torque=False)

    # Let the CAN interface and motor settle after bus shutdown/reopen
    print(f"\n{Fore.CYAN}Waiting for motor to settle...{Style.RESET_ALL}")
    time.sleep(1.0)

    # Step 5: Scan again to verify
    print(f"{Fore.CYAN}Verifying...{Style.RESET_ALL}")
    verified = scan_motors(channel)

    if verified and new_id in verified:
        print(f"\n{Fore.GREEN}Success! Motor is now responding on ID {Fore.WHITE}{Style.BRIGHT}{new_id}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}WARNING: Motor not found on new ID {new_id}. "
              f"It may require a power cycle to take effect.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
