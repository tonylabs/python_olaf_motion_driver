"""Change the CAN bus ID of a DaMiao motor.

Scans for connected motors, lets the user pick one and assign new IDs
(both receive/command ID and feedback ID), then scans again to confirm.

DaMiao actuators have two CAN IDs:
  - CAN_ID (Register 8): The ID the actuator listens on for commands
  - MASTER_ID (Register 7): The ID used in feedback/response frames

Usage:
    python damiao/chg_id.py [channel]

Example:
    python damiao/chg_id.py
    python damiao/chg_id.py can1
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from damiao_motor import DaMiaoController
from damiao_motor.core.motor import MOTOR_TYPE_PRESETS
from damiao_motor.cli.display import scan_motors

init(autoreset=True)

ACTUATOR_TYPES = sorted(MOTOR_TYPE_PRESETS.keys())


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


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else "can_usb"

    # Step 0: Select actuator type
    actuator_type = select_actuator_type()

    # Step 1: Scan for connected actuators
    print(f"\n{Fore.CYAN}Scanning {channel} for DaMiao actuators...{Style.RESET_ALL}")
    motor_ids = list(range(0x01, 0x20))
    responded = scan_motors(
        channel=channel, motor_ids=motor_ids, duration_s=3.0,
        motor_type=actuator_type,
    )

    if not responded:
        print(f"{Fore.RED}No actuators found.{Style.RESET_ALL}")
        return

    found_ids = sorted(responded)

    # Step 2: Show numbered list and let user pick
    print(f"\n{Fore.LIGHTYELLOW_EX}Available actuators:{Style.RESET_ALL}")
    for idx, mid in enumerate(found_ids, 1):
        print(f"  {Fore.WHITE}{Style.BRIGHT}{idx}{Style.RESET_ALL}) Actuator ID {Fore.WHITE}{Style.BRIGHT}0x{mid:02X} ({mid}){Style.RESET_ALL}")

    while True:
        raw = input(f"\nSelect actuator to change [1-{len(found_ids)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(found_ids):
                old_id = found_ids[choice - 1]
                break
            print(f"{Fore.RED}Please enter a number between 1 and {len(found_ids)}.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

    # Step 3: Ask for the new CAN_ID
    while True:
        raw = input(f"Enter the new CAN_ID for actuator 0x{old_id:02X} (1-127): ").strip()
        try:
            new_can_id = int(raw, 0)  # support 0x prefix
            if not (1 <= new_can_id <= 127):
                print(f"{Fore.RED}ID must be between 1 and 127.{Style.RESET_ALL}")
                continue
            if new_can_id == old_id:
                print(f"{Fore.RED}New ID is the same as the current ID (0x{old_id:02X}). Try again.{Style.RESET_ALL}")
                continue
            if new_can_id in found_ids:
                print(f"{Fore.RED}ID 0x{new_can_id:02X} is already in use by another actuator. Try again.{Style.RESET_ALL}")
                continue
            break
        except ValueError:
            print(f"{Fore.RED}Please enter a valid integer (e.g. 3 or 0x03).{Style.RESET_ALL}")

    # Step 4: Ask for the new MASTER_ID, default to 0x00
    raw = input(f"Enter the new MASTER_ID (default 0x00, press Enter to keep): ").strip()
    if raw:
        try:
            new_master_id = int(raw, 0)
        except ValueError:
            print(f"{Fore.LIGHTYELLOW_EX}Invalid input, using default 0x00{Style.RESET_ALL}")
            new_master_id = 0x00
    else:
        new_master_id = 0x00

    # Step 5: Change the IDs
    print(f"\n{Fore.CYAN}Changing actuator 0x{old_id:02X}:{Style.RESET_ALL}")
    print(f"  {Fore.LIGHTYELLOW_EX}CAN_ID (receive):    0x{old_id:02X} -> 0x{new_can_id:02X}{Style.RESET_ALL}")
    print(f"  {Fore.LIGHTYELLOW_EX}MASTER_ID (feedback): -> 0x{new_master_id:02X}{Style.RESET_ALL}")

    controller = DaMiaoController(channel=channel, bustype="socketcan")
    motor = controller.add_motor(
        motor_id=old_id, feedback_id=0x00, motor_type=actuator_type
    )

    try:
        # Read current register values first
        print(f"\n{Fore.CYAN}Reading current registers...{Style.RESET_ALL}")
        cur_can = motor.get_register(8, timeout=1.0)
        cur_master = motor.get_register(7, timeout=1.0)
        print(f"  Current CAN_ID (reg 8):    {Fore.WHITE}{Style.BRIGHT}{int(cur_can)}{Style.RESET_ALL}")
        print(f"  Current MASTER_ID (reg 7): {Fore.WHITE}{Style.BRIGHT}{int(cur_master)}{Style.RESET_ALL}")

        # Write MASTER_ID first, then CAN_ID last — changing CAN_ID
        # immediately changes what the actuator listens on, so all other
        # register writes must happen before it.
        print(f"\n{Fore.CYAN}Writing new IDs...{Style.RESET_ALL}")
        motor.set_feedback_id(new_master_id)
        print(f"  {Fore.LIGHTYELLOW_EX}MASTER_ID set to 0x{new_master_id:02X}{Style.RESET_ALL}")
        time.sleep(0.1)

        motor.set_receive_id(new_can_id)
        # Update motor object to use the new CAN_ID so store_parameters()
        # addresses the command correctly (data bytes include canid)
        motor.motor_id = new_can_id
        print(f"  {Fore.LIGHTYELLOW_EX}CAN_ID set to 0x{new_can_id:02X}{Style.RESET_ALL}")
        time.sleep(0.1)

        # Save to flash
        print(f"\n{Fore.CYAN}Saving to flash...{Style.RESET_ALL}")
        motor.store_parameters()
        time.sleep(0.5)
        print(f"  {Fore.GREEN}Parameters stored to flash.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}ERROR: {e}{Style.RESET_ALL}")
        controller.bus.shutdown()
        return

    controller.bus.shutdown()

    # Step 6: Scan again to verify
    print(f"\n{Fore.CYAN}Waiting for actuator to settle...{Style.RESET_ALL}")
    time.sleep(1.0)

    print(f"{Fore.CYAN}Verifying...{Style.RESET_ALL}")
    verified = scan_motors(
        channel=channel, motor_ids=motor_ids, duration_s=3.0,
        motor_type=actuator_type,
    )

    if verified and new_can_id in verified:
        print(f"\n{Fore.GREEN}Success! Actuator is now responding on CAN_ID {Fore.WHITE}{Style.BRIGHT}0x{new_can_id:02X} ({new_can_id}){Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}WARNING: Actuator not found on new CAN_ID 0x{new_can_id:02X}. "
              f"It may require a power cycle to take effect.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
