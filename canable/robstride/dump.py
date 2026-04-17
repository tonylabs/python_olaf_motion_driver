"""Quick test: scan for motors and hold position at 0 rad."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorama import Fore, Style, init
from lib.robstride import RobstrideBus, Motor

init(autoreset=True)

# Scan for motors on the bus
print(f"{Fore.CYAN}Scanning for motors...{Style.RESET_ALL}")
device_ids = RobstrideBus.scan_channel("can_usb")
print(f"{Fore.LIGHTYELLOW_EX}Found motors: {device_ids}{Style.RESET_ALL}")

if not device_ids:
    print(f"{Fore.RED}No motors found. Check wiring and power.{Style.RESET_ALL}")
    exit(1)

# Use the first motor found
motor_id = next(iter(device_ids))
print(f"{Fore.GREEN}Using motor ID: {Fore.WHITE}{Style.BRIGHT}{motor_id}{Style.RESET_ALL}")

motors = {"test": Motor(id=motor_id, model="rs-03")}
bus = RobstrideBus("can_usb", motors)
bus.connect()
bus.enable("test")

try:
    while True:
        bus.write_operation_frame(
            "test", position=0.0, kp=30.0, kd=0.5
        )
        pos, vel, torque, temp = bus.read_operation_frame("test")
        print(f"Position: {pos:+.3f} rad  Velocity: {vel:+.3f} rad/s  Torque: {torque:+.3f} Nm")
        time.sleep(0.01)
except KeyboardInterrupt:
    print(f"\n{Fore.LIGHTYELLOW_EX}Stopping...{Style.RESET_ALL}")
finally:
    bus.disconnect()
    print(f"{Fore.CYAN}Disconnected{Style.RESET_ALL}")
