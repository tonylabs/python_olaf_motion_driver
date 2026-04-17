"""Home all 12 actuators (Robstride + Damiao) to zero on a single CAN bus.

Actuator layout:
    ID 1  — Robstride RS-02
    ID 2  — Robstride RS-03
    ID 3  — Robstride RS-03
    ID 4  — Robstride RS-03
    ID 5  — Damiao DM-J4340P-2EC  (master_id=1, can_id=5)
    ID 6  — Damiao DM-J4310-2EC   (master_id=2, can_id=6)
    ID 7  — Robstride RS-02
    ID 8  — Robstride RS-03
    ID 9  — Robstride RS-03
    ID 10 — Robstride RS-03
    ID 11 — Damiao DM-J4340P-2EC  (master_id=3, can_id=11)
    ID 12 — Damiao DM-J4310-2EC   (master_id=4, can_id=12)

Usage:
    python all_homing.py [--speed 0.2] [--channel can_usb]
"""

import argparse
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorama import Fore, Style, init
from lib.robstride import RobstrideBus, Motor
from damiao_motor import DaMiaoController

init(autoreset=True)

# ── Actuator definitions ─────────────────────────────────────────────────────

ROBSTRIDE_MOTORS = [
    # (motor_id, model)
    (1,  "rs-02"),
    (2,  "rs-03"),
    (3,  "rs-03"),
    (4,  "rs-03"),
    (7,  "rs-02"),
    (8,  "rs-03"),
    (9,  "rs-03"),
    (10, "rs-03"),
]

DAMIAO_MOTORS = [
    # (master_id, can_id, motor_type)
    (1, 5,  "4340P"),
    (2, 6,  "4310"),
    (3, 11, "4340P"),
    (4, 12, "4310"),
]

# Per-model MIT gains
ROBSTRIDE_GAINS = {
    "rs-02": {"kp": 30.0, "kd": 5.0},
    "rs-03": {"kp": 30.0, "kd": 5.0},
}

DAMIAO_GAINS = {
    "4340P": {"kp": 100.0, "kd": 3.0},
    "4310":  {"kp": 100.0, "kd": 3.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Home all actuators to zero")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="Max homing speed in rad/s (default: 0.2)")
    parser.add_argument("--channel", default="can_usb",
                        help="CAN channel (default: can_usb)")
    args = parser.parse_args()

    dt = 0.01
    channel = args.channel

    # ── Set up Robstride bus ──────────────────────────────────────────────
    rs_motor_defs = {}
    for mid, model in ROBSTRIDE_MOTORS:
        name = f"rs_{mid}"
        rs_motor_defs[name] = Motor(id=mid, model=model)

    rs_bus = RobstrideBus(channel, rs_motor_defs)
    rs_bus.connect()

    # ── Set up Damiao controller ──────────────────────────────────────────
    if sys.platform == "darwin":
        dm_ctrl = DaMiaoController(
            channel="0046002E594E501820313332", bustype="gs_usb", bitrate=1000000
        )
    else:
        dm_ctrl = DaMiaoController(channel=channel, bustype="socketcan")

    # ── Data structures for all motors ────────────────────────────────────
    # Each entry: label, target, goal, and the send/read functions
    labels = []       # display labels for all motors
    targets = {}      # current ramp target (rad)
    goals = {}        # goal position (nearest 2π multiple)
    gains = {}        # (kp, kd) per motor
    settled = set()

    # -- Robstride: enable and read starting positions
    rs_labels = []
    rs_models = {}
    for mid, model in ROBSTRIDE_MOTORS:
        name = f"rs_{mid}"
        rs_bus.enable(name)
        rs_bus.write_operation_frame(name, position=0.0, kp=0.0, kd=0.0)
        start_pos, _, _, _ = rs_bus.read_operation_frame(name)
        nearest_zero = round(start_pos / (2 * math.pi)) * (2 * math.pi)

        label = f"RS:{mid}"
        labels.append(label)
        rs_labels.append(label)
        rs_models[label] = (name, model)
        targets[label] = start_pos
        goals[label] = nearest_zero
        g = ROBSTRIDE_GAINS[model]
        gains[label] = (g["kp"], g["kd"])
        distance = start_pos - nearest_zero
        print(
            f"{Fore.LIGHTYELLOW_EX}  {label:<8} ({model})  "
            f"pos={start_pos:+.3f}  goal={nearest_zero:+.3f}  "
            f"distance={distance:+.3f} rad{Style.RESET_ALL}"
        )

    # -- Damiao: enable and read starting positions
    dm_labels = []
    dm_motors = {}
    for master_id, can_id, mtype in DAMIAO_MOTORS:
        motor = dm_ctrl.add_motor(
            motor_id=can_id, feedback_id=master_id, motor_type=mtype
        )
        motor.enable()
        motor.send_cmd_mit(
            target_position=0.0, target_velocity=0.0,
            stiffness=0.0, damping=0.0, feedforward_torque=0.0,
        )
        time.sleep(0.05)
        dm_ctrl.poll_feedback()
        start_pos = motor.state.get("pos", 0.0)
        nearest_zero = round(start_pos / (2 * math.pi)) * (2 * math.pi)

        label = f"DM:{master_id}:{can_id}"
        labels.append(label)
        dm_labels.append(label)
        dm_motors[label] = (motor, mtype)
        targets[label] = start_pos
        goals[label] = nearest_zero
        g = DAMIAO_GAINS[mtype]
        gains[label] = (g["kp"], g["kd"])
        distance = start_pos - nearest_zero
        print(
            f"{Fore.LIGHTYELLOW_EX}  {label:<8} ({mtype})  "
            f"pos={start_pos:+.3f}  goal={nearest_zero:+.3f}  "
            f"distance={distance:+.3f} rad{Style.RESET_ALL}"
        )

    total = len(labels)
    print(f"\n{Fore.CYAN}Homing {total} actuator(s) to zero at {args.speed} rad/s...{Style.RESET_ALL}")

    # ── Main control loop ─────────────────────────────────────────────────
    try:
        step = args.speed * dt
        max_distance = max(abs(targets[l] - goals[l]) for l in labels)
        max_iters = int(max_distance / step) + 1000

        for i in range(max_iters):
            # Ramp all targets toward their goals
            for label in labels:
                goal = goals[label]
                diff = targets[label] - goal
                if diff > step:
                    targets[label] -= step
                elif diff < -step:
                    targets[label] += step
                else:
                    targets[label] = goal

            # Send commands to all Robstride motors
            for label in rs_labels:
                name, model = rs_models[label]
                kp, kd = gains[label]
                rs_bus.write_operation_frame(
                    name, position=targets[label], kp=kp, kd=kd
                )
                pos, vel, torque, temp = rs_bus.read_operation_frame(name)

                if label not in settled:
                    if i % 100 == 0:
                        print(
                            f"  [{label}] target={targets[label]:+.3f}  "
                            f"pos={pos:+.3f}  vel={vel:+.3f}  torque={torque:+.3f}"
                        )
                    if targets[label] == goals[label] and abs(pos - goals[label]) < 0.02 and abs(vel) < 0.3:
                        print(f"  {Fore.GREEN}[{label}] Reached zero (pos={pos:+.4f}){Style.RESET_ALL}")
                        settled.add(label)

            # Send commands to all Damiao motors
            for label in dm_labels:
                motor, mtype = dm_motors[label]
                kp, kd = gains[label]
                motor.send_cmd_mit(
                    target_position=targets[label],
                    target_velocity=0.0,
                    stiffness=kp,
                    damping=kd,
                    feedforward_torque=0.0,
                )
                dm_ctrl.poll_feedback()
                pos = motor.state.get("pos", 0.0)
                vel = motor.state.get("vel", 0.0)
                torq = motor.state.get("torq", 0.0)

                if label not in settled:
                    if i % 100 == 0:
                        print(
                            f"  [{label}] target={targets[label]:+.3f}  "
                            f"pos={pos:+.3f}  vel={vel:+.3f}  torque={torq:+.3f}"
                        )
                    if targets[label] == goals[label] and abs(pos - goals[label]) < 0.15 and abs(vel) < 0.3:
                        print(f"  {Fore.GREEN}[{label}] Reached zero (pos={pos:+.4f}){Style.RESET_ALL}")
                        settled.add(label)

            if len(settled) == total:
                # Hold all briefly to stabilize
                for _ in range(100):
                    for label in rs_labels:
                        name, _ = rs_models[label]
                        kp, kd = gains[label]
                        rs_bus.write_operation_frame(name, position=goals[label], kp=kp, kd=kd)
                        rs_bus.read_operation_frame(name)
                    for label in dm_labels:
                        motor, _ = dm_motors[label]
                        kp, kd = gains[label]
                        motor.send_cmd_mit(
                            target_position=goals[label], target_velocity=0.0,
                            stiffness=kp, damping=kd, feedforward_torque=0.0,
                        )
                        dm_ctrl.poll_feedback()
                    time.sleep(dt)
                break

            time.sleep(dt)
        else:
            not_settled = [l for l in labels if l not in settled]
            print(f"  {Fore.RED}Timed out. Not settled: {not_settled}{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}Homing complete.{Style.RESET_ALL}")

    except KeyboardInterrupt:
        print(f"\n{Fore.LIGHTYELLOW_EX}Interrupted.{Style.RESET_ALL}")
    finally:
        dm_ctrl.disable_all()
        rs_bus.disconnect()
        try:
            dm_ctrl.bus.shutdown()
        except Exception:
            pass
        print(f"{Fore.CYAN}All actuators disconnected.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()