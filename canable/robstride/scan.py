"""Scan the CAN bus for Robstride motors.

Uses the library's scan_channel but patches the transmit method to handle
'Transmit buffer full' errors with retry/backoff.

Usage:
    python robstride/scan.py [channel] [start_id] [end_id]
"""
import sys
import can
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colorama import Fore, Style, init
from lib.robstride import RobstrideBus

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


if __name__ == "__main__":
    channel = sys.argv[1] if len(sys.argv) > 1 else "can_usb"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    print(f"{Fore.CYAN}Scanning {channel} for IDs {start}-{end-1}...{Style.RESET_ALL}")
    motors = RobstrideBus.scan_channel(channel, start, end)

    if motors:
        print(f"\n{Fore.GREEN}Found {len(motors)} motor(s):{Style.RESET_ALL}")
        for motor_id, info in motors.items():
            print(f"  {Fore.GREEN}Motor ID: {Fore.WHITE}{Style.BRIGHT}{motor_id}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}No motors found.{Style.RESET_ALL}")
