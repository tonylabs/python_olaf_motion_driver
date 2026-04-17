# OLAF Raspberry Pi Runtime Driver

Skeleton for deploying a skrl-trained walking policy to a physical OLAF via a USB-CAN adapter at 1 Mbps.

## Layout

| File                | Purpose                                                      |
|---------------------|--------------------------------------------------------------|
| `config.py`         | Joint order, default targets, motor types, CAN IDs, limits   |
| `can_bus.py`        | SocketCAN wrapper over `python-can`                          |
| `motors.py`         | Robstride RS02/RS03 and Damiao DM-J43xx MIT-mode drivers     |
| `imu.py`            | IMU interface (replace stub with your board's driver)        |
| `observation.py`    | Observation assembly + path-frame tracker                    |
| `policy.py`         | ONNX policy + RunningStandardScaler re-implementation        |
| `run.py`            | Main entry — 50 Hz policy tick + 600 Hz motor tick           |

## Setup on the Pi

```bash
# 1. CAN bring-up (SocketCAN-compatible adapter)
sudo ip link set can0 up type can bitrate 1000000
sudo ifconfig can0 txqueuelen 1000

# 2. Python deps
pip install onnxruntime python-can numpy pyyaml
```

## Workflow

```bash
# On training machine — export policy + preprocessor
uv sync
python export_policy.py --checkpoint policy.pt --task Olaf-Bipedal-Walking-v0 --out driver/deploy/
python run.py --policy-dir deploy/
```

## What you MUST verify before enabling torque

1. **Joint order**: `config.JOINT_ORDER` must exactly match the training env's
   `LEG_JOINT_NAMES`. A swapped index looks almost-working then diverges.
2. **CAN IDs**: fill in `config.MOTOR_TABLE` with your wiring.
3. **IMU frame convention**: observation uses gravity projected into the root
   frame and angular velocity in the root frame. Make sure your IMU mount
   orientation matches the URDF's `base_link` — re-express if not.
4. **Obs concat order**: must match training exactly
   (velocity_cmd, root_pose_in_path, base_lin_vel, base_ang_vel,
    projected_gravity, joint_pos_rel, joint_vel, last_action,
    prev_prev_action, gait_phase). Print sim obs and Pi obs for the same
   canned pose and diff them.
5. **Running-scaler stats**: exported JSON contains mean/var — applied in
   `policy.py`. If skipped, policy sees inputs with wrong scale and fails.

## Safety ramp

`run.py` boots motors with kp=0, kd=small, ramps kp over ~2 s before handing
control to the policy. Watchdog drops to damping if a policy tick takes
longer than 40 ms. Hardware E-stop on the motor power rail is mandatory — do
not rely on CAN disable.

## Debug flags

Three flags on `run.py` for bring-up work. They compose — `--slomo --joints …`
is the normal way to first move a single joint under the policy.

| Flag | Purpose |
|------|---------|
| `--slomo` | Clamp per-joint target slew to `config.SLOMO_VMAX_RAD_S` (default 0.5 rad/s) before publishing to the motor thread. Applied after the per-joint active mask so the clamp sees the delta the motors will actually execute. |
| `--joints NAMES` | Comma-separated joint names or indices (e.g. `l_hip_yaw,r_knee_pitch` or `0,9`). Listed joints track the policy; all others hold at `DEFAULT_JOINT_POS` with normal kp (robot stays propped up). Names accept the `_joint` suffix or the short form. Default: all 12. |
| `--log-level` | Standard `logging` level. Use `INFO` (default) for normal runs — `DEBUG` floods the terminal with per-frame SocketCAN traces. |

Examples:

```bash
python run.py --slomo --joints l_knee_pitch                  # one joint, slowly
python run.py --joints l_ankle_pitch,l_ankle_roll,r_ankle_pitch,r_ankle_roll
python run.py --slomo                                        # all joints, slow
```

### Non-finite guards

Two guards prevent a single bad serial frame or NaN from the policy from
crashing the motor thread:

- **IMU (`imu.py`)**: quaternion with zero/non-finite/out-of-range norm falls
  back to `(0, 0, −1)` projected gravity; non-finite angular velocity is
  zeroed. Logs a warning.
- **Motor loop (`run.py`)**: if the interpolated target is non-finite, the
  LPF is reset to the measured joint positions and the tick falls through to
  `damp_all()` — same policy as the watchdog path.

If you see `non-finite IMU angular velocity` or `non-finite target, damping
this tick` warnings, the IMU serial parser is desyncing; the loop will keep
running but the root cause is upstream.