# Reinforcement Learning Guide: Bipedal Robot with Robstride & Damiao Actuators

A practical guide for training a bipedal robot to stand, march in place (原地踏步), and walk with a playful Disney Olaf-style waddle gait using RL — built on top of this project's actuator stack.

---

## Table of Contents

1. [Hardware Layout](#1-hardware-layout)
2. [Software Stack](#2-software-stack)
3. [Simulation Environment](#3-simulation-environment)
4. [Observation & Action Spaces](#4-observation--action-spaces)
5. [Phase 1 — Standing Balance](#5-phase-1--standing-balance)
6. [Phase 2 — Marching in Place (原地踏步)](#6-phase-2--marching-in-place-原地踏步)
7. [Phase 3 — Olaf-Style Walking](#7-phase-3--olaf-style-walking)
8. [Sim-to-Real Transfer](#8-sim-to-real-transfer)
9. [Real-World Deployment](#9-real-world-deployment)
10. [Safety](#10-safety)
11. [References](#11-references)

---

## 1. Hardware Layout

A 12-DoF bipedal using the same motor assignment as `all_homing.py`:

```
                 ┌──────────┐
                 │   Torso   │
                 │  (IMU)    │
                 └────┬─────┘
            ┌─────────┴─────────┐
       Left Leg              Right Leg
   ┌────────────┐         ┌────────────┐
   │ ID 1  Hip-Y│ RS-02   │ ID 7  Hip-Y│ RS-02      (yaw / rotation)
   │ ID 2  Hip-R│ RS-03   │ ID 8  Hip-R│ RS-03      (roll / abduction)
   │ ID 3  Hip-P│ RS-03   │ ID 9  Hip-P│ RS-03      (pitch / flexion)
   │ ID 4  Knee │ RS-03   │ ID 10 Knee │ RS-03      (pitch / flexion)
   │ ID 5  Ank-P│ DM-4340P│ ID 11 Ank-P│ DM-4340P   (pitch / flexion)
   │ ID 6  Ank-R│ DM-4310 │ ID 12 Ank-R│ DM-4310    (roll / inversion)
   └────────────┘         └────────────┘
```

**Why this mix?** RS-03 (60 Nm, 50 rad/s) handles high-torque hip and knee joints. DM-J4340P (28 Nm, 8 rad/s) is compact enough for ankle pitch. DM-J4310 (10 Nm, 30 rad/s) is lightweight for ankle roll. RS-02 handles hip yaw where moderate torque is sufficient.

**Additional sensors required:**
- **IMU** (e.g. BNO085 or MPU-6050) mounted on the torso — provides roll, pitch, yaw and angular velocity
- **Optional:** Foot pressure sensors (FSR) under each foot for contact detection

### IMU in URDF (SolidWorks Export)

The IMU is rigidly attached to the torso — it does not need a moving joint. When exporting URDF from SolidWorks using the SW2URDF plugin:

1. Model the IMU as a simple part (small box matching BNO085/MPU-6050 dimensions), mated to the torso body
2. In the URDF export tree, add it as a **child link of the torso** with a **fixed joint**
3. Set the joint origin to match the IMU's actual mounting position and orientation

SW2URDF does not generate sensor tags, so add the `imu_link` manually after export if needed:

```xml
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.05" rpy="0 0 0"/>  <!-- adjust to actual mount position -->
</joint>
```

**Important:** Make sure the IMU link frame axes match the physical IMU mounting orientation (X-forward, Z-up), so sim and real readings align during sim-to-real transfer.

---

## 2. Software Stack

```
┌─────────────────────────────────────────────────┐
│  RL Training (GPU workstation)                  │
│  ┌───────────────┐  ┌────────────────────────┐  │
│  │ Isaac Gym /    │  │ PPO / SAC Policy       │  │
│  │ MuJoCo + DM   │  │ (stable-baselines3 /   │  │
│  │ Simulation     │  │  RSL-RL / legged_gym)  │  │
│  └───────┬───────┘  └──────────┬─────────────┘  │
│          └──────────┬──────────┘                 │
│                     │ Trained policy (.pt)       │
└─────────────────────┼───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  Real Robot (Raspberry Pi / Jetson)              │
│  ┌───────────────┐  ┌────────────────────────┐  │
│  │ Policy         │  │ This project's API     │  │
│  │ Inference      │→ │ RobstrideBus +         │  │
│  │ (PyTorch CPU)  │  │ DaMiaoController       │  │
│  └───────────────┘  └──────────┬─────────────┘  │
│                                │ CAN bus         │
│                                ▼                 │
│                      12 Actuators on can0        │
└─────────────────────────────────────────────────┘
```

### Dependencies

```bash
# Simulation & Training
pip install mujoco gymnasium stable-baselines3 torch tensorboard

# Or use NVIDIA Isaac Gym for massively parallel training
# See: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

# Robot-specific RL framework (recommended)
pip install rsl-rl  # ETH Zurich's legged locomotion RL library

# Real-world deployment (on the robot's computer)
uv sync  # This project's actuator dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Simulation Environment

### 3.1 URDF / MJCF Model

Create a URDF model that matches the physical robot. Each joint must map to a real actuator:

```xml
<!-- biped.urdf (simplified) -->
<robot name="olaf_biped">
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left leg chain -->
  <joint name="l_hip_yaw" type="revolute">   <!-- ID 1, RS-02 -->
    <limit lower="-1.57" upper="1.57" effort="17" velocity="44"/>
  </joint>
  <joint name="l_hip_roll" type="revolute">  <!-- ID 2, RS-03 -->
    <limit lower="-0.5" upper="0.5" effort="60" velocity="50"/>
  </joint>
  <joint name="l_hip_pitch" type="revolute"> <!-- ID 3, RS-03 -->
    <limit lower="-1.57" upper="1.57" effort="60" velocity="50"/>
  </joint>
  <joint name="l_knee" type="revolute">      <!-- ID 5, DM-4340P -->
    <limit lower="0.0" upper="2.5" effort="28" velocity="8"/>
  </joint>
  <joint name="l_ankle_pitch" type="revolute"> <!-- ID 6, DM-4310 -->
    <limit lower="-0.8" upper="0.8" effort="10" velocity="30"/>
  </joint>
  <joint name="l_ankle_roll" type="revolute">  <!-- ID 4, RS-03 -->
    <limit lower="-0.5" upper="0.5" effort="60" velocity="50"/>
  </joint>

  <!-- Right leg: mirror of left (IDs 7-12) -->
  <!-- ... -->
</robot>
```

> **Key:** Match joint limits and torque limits to the real actuator specs from `robstride/lib/table.py` and the Damiao datasheet. Inaccurate limits are the #1 cause of sim-to-real failure.

### 3.2 MuJoCo Environment Skeleton

```python
# env/biped_env.py
import mujoco
import numpy as np
from gymnasium import Env, spaces

class BipedEnv(Env):
    JOINT_NAMES = [
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch",
        "l_knee", "l_ankle_pitch", "l_ankle_roll",
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch",
        "r_knee", "r_ankle_pitch", "r_ankle_roll",
    ]
    NUM_JOINTS = 12
    DT = 0.01  # 100 Hz control, matches real CAN loop

    def __init__(self, task="stand"):
        self.model = mujoco.MjModel.from_xml_path("biped.xml")
        self.data = mujoco.MjData(self.model)
        self.task = task

        # Action: target position offsets for each joint (rad)
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(self.NUM_JOINTS,), dtype=np.float32
        )
        # Observation: joint pos + vel + IMU orientation + angular vel
        obs_dim = self.NUM_JOINTS * 2 + 3 + 3 + 3  # pos, vel, rpy, gyro, command
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Default standing pose (slightly bent knees)
        self.default_pose = np.zeros(self.NUM_JOINTS)
        self.default_pose[3] = 0.4   # l_knee
        self.default_pose[9] = 0.4   # r_knee
        self.default_pose[2] = -0.2  # l_hip_pitch
        self.default_pose[8] = -0.2  # r_hip_pitch

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        # Randomize initial state slightly for robustness
        self.data.qpos[:self.NUM_JOINTS] = self.default_pose + \
            np.random.uniform(-0.05, 0.05, self.NUM_JOINTS)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # Actions are position offsets from default pose
        target = self.default_pose + action
        # Simulate PD control at the joint level (mirrors MIT mode on real hw)
        kp = np.array([30, 30, 30, 100, 100, 30] * 2, dtype=np.float32)
        kd = np.array([5, 5, 5, 3, 3, 5] * 2, dtype=np.float32)
        pos_error = target - self.data.qpos[:self.NUM_JOINTS]
        vel = self.data.qvel[:self.NUM_JOINTS]
        self.data.ctrl[:] = kp * pos_error - kd * vel

        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_fallen()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        joint_pos = self.data.qpos[:self.NUM_JOINTS]
        joint_vel = self.data.qvel[:self.NUM_JOINTS]
        # Torso orientation (roll, pitch, yaw) from quaternion
        torso_quat = self.data.xquat[1]  # torso body
        rpy = quat_to_euler(torso_quat)
        gyro = self.data.sensordata[:3]  # gyroscope
        command = np.array([0.0, 0.0, 0.0])  # [vx_cmd, vy_cmd, yaw_rate_cmd]
        return np.concatenate([joint_pos, joint_vel, rpy, gyro, command])

    def _is_fallen(self):
        torso_height = self.data.xpos[1][2]  # torso z position
        return torso_height < 0.3  # fallen if torso below 30cm
```

### 3.3 Isaac Gym (Recommended for Speed)

For parallel training of thousands of environments on GPU, use [legged_gym](https://github.com/leggedrobotics/legged_gym) from ETH Zurich's RSL:

```bash
git clone https://github.com/leggedrobotics/legged_gym
cd legged_gym
pip install -e .
```

Create a config for the biped in `legged_gym/envs/biped/`:
- `biped_config.py` — hyperparameters, reward scales, domain randomization
- `biped.py` — environment class inheriting from `LeggedRobot`

---

## 4. Observation & Action Spaces

### Observations (input to the policy)

| Component | Dimension | Source |
|-----------|-----------|--------|
| Joint positions (normalized) | 12 | `read_operation_frame()` position / Damiao `motor.state["pos"]` |
| Joint velocities | 12 | `read_operation_frame()` velocity / Damiao `motor.state["vel"]` |
| Torso roll, pitch | 2 | IMU (gravity-referenced) |
| Torso angular velocity | 3 | IMU gyroscope |
| Velocity command | 3 | User input: `[vx, vy, yaw_rate]` |
| Previous action | 12 | From last timestep |
| **Total** | **44** | |

### Actions (output from the policy)

The policy outputs **12 position offsets** (in radians) added to a default standing pose. These are sent to actuators as position targets via MIT mode:

```python
target_position = default_pose[i] + action[i] * action_scale
# Then sent via:
bus.write_operation_frame(motor, position=target_position, kp=kp, kd=kd)
```

This is safer than direct torque control — the PD gains in MIT mode act as a compliance layer.

---

## 5. Phase 1 — Standing Balance

**Goal:** The robot maintains an upright standing posture and resists perturbations.

### Reward Function

```python
def reward_standing(self):
    # 1. Upright bonus: reward small torso tilt
    roll, pitch = self.torso_rpy[0], self.torso_rpy[1]
    r_upright = np.exp(-10 * (roll**2 + pitch**2))

    # 2. Height maintenance: reward being at target height
    r_height = np.exp(-40 * (self.torso_height - TARGET_HEIGHT)**2)

    # 3. Minimal joint velocity: penalize unnecessary motion
    r_still = np.exp(-0.1 * np.sum(self.joint_vel**2))

    # 4. Energy efficiency: penalize high torques
    r_energy = -0.001 * np.sum(self.joint_torque**2)

    # 5. Joint limit penalty
    r_limits = -10.0 * np.sum(np.maximum(
        np.abs(self.joint_pos) - self.joint_limits * 0.9, 0
    ))

    return 0.4 * r_upright + 0.3 * r_height + 0.2 * r_still + r_energy + r_limits
```

### Training

```python
from stable_baselines3 import PPO

env = BipedEnv(task="stand")
model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=1,
    tensorboard_log="./tb_logs/stand",
)
model.learn(total_timesteps=5_000_000)
model.save("policies/stand_v1")
```

### Domain Randomization (for robustness)

```python
# Randomize at each episode reset:
randomization = {
    "mass":      (+/- 15%),   # Total robot mass
    "friction":  (0.4, 1.2),  # Ground friction coefficient
    "kp_scale":  (0.8, 1.2),  # PD gain variation
    "kd_scale":  (0.8, 1.2),
    "push_force": (0, 30),    # Random external pushes (N)
    "latency":   (0, 20),     # Observation delay (ms)
    "noise":     (+/- 0.01),  # Sensor noise on joint pos (rad)
}
```

### Success Criteria

- Stands for 30+ seconds without falling
- Recovers from 20 N lateral pushes
- Torso tilt stays within +/- 5 degrees

---

## 6. Phase 2 — Marching in Place (原地踏步)

**Goal:** Alternating leg lifts without forward displacement — the fundamental stepping pattern.

### Reward Function

```python
def reward_marching(self):
    # Phase clock: generates alternating left/right swing targets
    phase = (self.timestep * self.DT) % self.gait_period
    swing_left = phase < self.gait_period / 2

    # 1. Foot clearance: reward the swing foot being lifted
    l_foot_z = self.get_foot_height("left")
    r_foot_z = self.get_foot_height("right")
    target_lift = 0.05  # 5 cm lift height

    if swing_left:
        r_swing = np.exp(-100 * (l_foot_z - target_lift)**2)
        r_stance = np.exp(-100 * r_foot_z**2)  # right foot on ground
    else:
        r_swing = np.exp(-100 * (r_foot_z - target_lift)**2)
        r_stance = np.exp(-100 * l_foot_z**2)

    # 2. No lateral drift
    r_no_drift = np.exp(-5 * (self.base_xy_displacement**2).sum())

    # 3. Upright (same as standing)
    roll, pitch = self.torso_rpy[0], self.torso_rpy[1]
    r_upright = np.exp(-10 * (roll**2 + pitch**2))

    # 4. Rhythmicity: reward consistent stepping frequency
    r_rhythm = np.exp(-2 * (self.step_frequency - TARGET_FREQ)**2)

    # 5. Symmetric motion
    l_joints = self.joint_pos[:6]
    r_joints = self.joint_pos[6:]
    # At half-phase offset, left and right should be mirrored
    r_symmetry = np.exp(-5 * np.sum(
        (l_joints - self._mirror(r_joints, phase))**2
    ))

    return (0.25 * r_swing + 0.15 * r_stance + 0.2 * r_no_drift +
            0.2 * r_upright + 0.1 * r_rhythm + 0.1 * r_symmetry)
```

### Curriculum

```
Epoch    0-500k:  Low lift (2 cm), slow cadence (0.5 Hz)
Epoch  500k-2M:   Medium lift (5 cm), normal cadence (1.0 Hz)
Epoch  2M-5M:     Full lift (5 cm), add perturbations
```

### Success Criteria

- Alternating foot lifts for 60+ seconds
- Center of mass stays within 10 cm of start position
- Stable cadence of 1.0 Hz (+/- 0.1 Hz)

---

## 7. Phase 3 — Olaf-Style Walking

**Goal:** A forward-moving gait with the characteristic Olaf waddle — wide stance, exaggerated lateral sway, slightly bouncy, arms-free torso bobbing. Cute and expressive, not energy-optimal.

### What Makes It "Olaf"

Disney's Olaf walks with:
- **Wide lateral sway** — the torso rocks side-to-side with each step
- **Short, quick steps** — high cadence, short stride length
- **Bouncy vertical motion** — slight up-down bob per step
- **Slightly turned-out feet** — toe-out angle giving a penguin-like quality
- **Joyful, carefree vibe** — not stiff or robotic

### Reward Function

```python
def reward_olaf_walk(self):
    # ── Forward progress ──
    # Command velocity: slow forward (~0.3 m/s), no lateral, no turn
    vx_cmd, vy_cmd, yaw_cmd = self.command_velocity
    vx = self.base_lin_vel[0]
    vy = self.base_lin_vel[1]
    r_forward = np.exp(-4 * (vx - vx_cmd)**2)
    r_lateral_track = np.exp(-4 * (vy - vy_cmd)**2)

    # ── Olaf character rewards ──

    # 1. Lateral sway: ENCOURAGE torso roll oscillation (not penalize it!)
    #    Target: +/- 8 degrees of roll, synchronized with gait phase
    phase = (self.timestep * self.DT) % self.gait_period
    target_roll = 0.14 * np.sin(2 * np.pi * phase / self.gait_period)  # ~8 deg
    r_sway = np.exp(-30 * (self.torso_rpy[0] - target_roll)**2)

    # 2. Vertical bounce: slight up-down synchronized with double-stance
    target_bounce = TARGET_HEIGHT + 0.015 * np.cos(
        4 * np.pi * phase / self.gait_period
    )  # 1.5 cm bounce at 2x step freq
    r_bounce = np.exp(-60 * (self.torso_height - target_bounce)**2)

    # 3. Toe-out angle: reward slight external rotation of feet
    l_yaw = self.joint_pos[0]  # l_hip_yaw
    r_yaw = self.joint_pos[6]  # r_hip_yaw
    target_toe_out = 0.15  # ~8.5 degrees outward
    r_toe_out = np.exp(-20 * ((l_yaw - target_toe_out)**2 +
                               (r_yaw + target_toe_out)**2))

    # 4. Short stride: penalize large hip pitch amplitude
    l_hip_pitch = self.joint_pos[2]
    r_hip_pitch = self.joint_pos[8]
    hip_amplitude = max(abs(l_hip_pitch), abs(r_hip_pitch))
    r_short_stride = np.exp(-5 * max(hip_amplitude - 0.4, 0)**2)

    # 5. Quick cadence: reward 1.5 Hz stepping (faster than normal walk)
    r_cadence = np.exp(-3 * (self.step_frequency - 1.5)**2)

    # ── Safety constraints (still important) ──
    pitch = self.torso_rpy[1]
    r_no_fall = np.exp(-10 * pitch**2)  # Don't fall forward/backward
    r_energy = -0.0005 * np.sum(self.joint_torque**2)

    # ── Smoothness ──
    r_smooth = -0.01 * np.sum((self.action - self.prev_action)**2)

    return (
        0.20 * r_forward +
        0.05 * r_lateral_track +
        0.15 * r_sway +          # Olaf lateral rock
        0.10 * r_bounce +        # Olaf bounce
        0.10 * r_toe_out +       # Penguin toe-out
        0.08 * r_short_stride +  # Short steps
        0.07 * r_cadence +       # Quick feet
        0.15 * r_no_fall +
        r_energy +
        r_smooth
    )
```

### Gait Parameters

| Parameter | Normal Walk | Olaf Walk |
|-----------|-------------|-----------|
| Speed | 0.5 m/s | 0.2-0.3 m/s |
| Cadence | 1.0 Hz | 1.5 Hz |
| Stride length | 0.5 m | 0.15-0.2 m |
| Lateral sway | +/- 2 deg | +/- 8 deg |
| Vertical bounce | ~0 cm | +/- 1.5 cm |
| Toe-out angle | 0 deg | ~8 deg |
| Foot clearance | 3 cm | 4-5 cm |

### Curriculum Strategy

```
Stage 1 (0-2M steps):
  - Initialize from Phase 2 (marching) policy
  - Low forward speed command (0.1 m/s)
  - No Olaf style rewards yet — just learn to walk forward

Stage 2 (2M-5M steps):
  - Increase forward speed to 0.2 m/s
  - Enable sway and bounce rewards (low weight: 0.05 each)
  - Enable toe-out reward

Stage 3 (5M-10M steps):
  - Full Olaf reward weights as shown above
  - Forward speed 0.3 m/s
  - Add perturbations and domain randomization
  - Fine-tune cadence and stride length
```

### Training Tips

- **Use the Phase 2 policy as initialization** — transfer learning dramatically speeds up convergence
- **Gait phase clock is essential** — without it, the policy learns to shuffle or hop instead of alternating steps
- **Don't penalize lateral sway** — this is the opposite of normal walking RL; Olaf's charm comes from the waddle
- **Tune `r_sway` carefully** — too much sway and the robot falls, too little and it looks boring

---

## 8. Sim-to-Real Transfer

### 8.1 Actuator Model in Simulation

The real actuators have delays, friction, and bandwidth limits. Model them:

```python
class ActuatorModel:
    """Simulates MIT-mode PD control with realistic actuator dynamics."""

    def __init__(self, kp, kd, torque_limit, velocity_limit, latency_steps=2):
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit
        self.velocity_limit = velocity_limit
        self.latency_steps = latency_steps
        self.cmd_buffer = []  # ring buffer for latency

    def compute_torque(self, target_pos, current_pos, current_vel):
        # Delayed command (simulates CAN bus + processing latency)
        self.cmd_buffer.append(target_pos)
        if len(self.cmd_buffer) > self.latency_steps:
            delayed_target = self.cmd_buffer.pop(0)
        else:
            delayed_target = self.cmd_buffer[0]

        torque = self.kp * (delayed_target - current_pos) - self.kd * current_vel
        return np.clip(torque, -self.torque_limit, self.torque_limit)
```

### 8.2 Domain Randomization Ranges

Based on real hardware characterization:

```python
DOMAIN_RANDOMIZATION = {
    # Actuator-specific
    "kp_scale":          (0.75, 1.25),   # PD gain mismatch
    "kd_scale":          (0.75, 1.25),
    "torque_noise":      0.5,            # Nm, random per-step
    "motor_strength":    (0.85, 1.15),   # torque output scaling
    "friction":          (0.5, 2.0),     # joint coulomb friction

    # Communication
    "action_latency_ms": (5, 25),        # CAN bus round-trip
    "obs_latency_ms":    (5, 15),

    # Mechanical
    "mass_scale":        (0.85, 1.15),
    "com_offset":        0.02,           # m, center of mass perturbation
    "link_length_scale": (0.98, 1.02),

    # Environment
    "ground_friction":   (0.4, 1.2),
    "push_magnitude":    (0, 40),        # N, random impulse
    "push_interval":     (3.0, 8.0),     # seconds between pushes

    # Sensor noise
    "joint_pos_noise":   0.01,           # rad
    "joint_vel_noise":   0.5,            # rad/s
    "imu_noise":         0.02,           # rad for orientation
    "gyro_noise":        0.1,            # rad/s
}
```

### 8.3 Observation Filtering

On real hardware, apply exponential moving average to noisy sensor readings:

```python
class ObservationFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev = None

    def __call__(self, obs):
        if self.prev is None:
            self.prev = obs.copy()
        self.prev = self.alpha * obs + (1 - self.alpha) * self.prev
        return self.prev.copy()
```

---

## 9. Real-World Deployment

### 9.1 Policy Inference Loop

```python
"""Deploy trained policy on real robot at 100 Hz."""

import time
import math
import numpy as np
import torch
from lib.robstride import RobstrideBus, Motor
from damiao_motor import DaMiaoController

# ── Load trained policy ──
device = torch.device("cpu")
policy = torch.jit.load("policies/olaf_walk.pt", map_location=device)
policy.eval()

# ── Motor setup (same as all_homing.py) ──
ROBSTRIDE_MOTORS = [
    (1, "rs-02"), (2, "rs-03"), (3, "rs-03"), (4, "rs-03"),
    (7, "rs-02"), (8, "rs-03"), (9, "rs-03"), (10, "rs-03"),
]
DAMIAO_MOTORS = [
    (1, 5, "4340P"), (2, 6, "4310"),
    (3, 11, "4340P"), (4, 12, "4310"),
]

# Joint order for the policy (must match training)
JOINT_ORDER = [
    ("rs", 1), ("rs", 2), ("rs", 3),  # L hip yaw, roll, pitch
    ("dm", 5), ("dm", 6), ("rs", 4),  # L knee, ankle pitch, ankle roll
    ("rs", 7), ("rs", 8), ("rs", 9),  # R hip yaw, roll, pitch
    ("dm", 11), ("dm", 12), ("rs", 10)  # R knee, ankle pitch, ankle roll
]

# PD gains per joint (same as training)
KP = np.array([30, 30, 30, 100, 100, 30, 30, 30, 30, 100, 100, 30], dtype=np.float32)
KD = np.array([5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3, 5], dtype=np.float32)

# Default standing pose
DEFAULT_POSE = np.zeros(12)
DEFAULT_POSE[[3, 9]] = 0.4  # knees
DEFAULT_POSE[[2, 8]] = -0.2  # hip pitch

ACTION_SCALE = 0.3  # rad, max deviation from default


def read_imu():
    """Read IMU — implement for your hardware (BNO085, MPU6050, etc.)."""
    # Returns: (roll, pitch, yaw, gyro_x, gyro_y, gyro_z)
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    # Initialize buses
    rs_defs = {f"rs_{mid}": Motor(id=mid, model=model)
               for mid, model in ROBSTRIDE_MOTORS}
    rs_bus = RobstrideBus("can0", rs_defs)
    rs_bus.connect()

    dm_ctrl = DaMiaoController(channel="can0", bustype="socketcan")
    dm_motors = {}
    for master_id, can_id, mtype in DAMIAO_MOTORS:
        m = dm_ctrl.add_motor(motor_id=can_id, feedback_id=master_id, motor_type=mtype)
        m.enable()
        dm_motors[can_id] = m

    # Enable all Robstride motors
    for name in rs_defs:
        rs_bus.enable(name)

    obs_filter = ObservationFilter(alpha=0.2)
    prev_action = np.zeros(12)
    command_velocity = np.array([0.3, 0.0, 0.0])  # vx=0.3 m/s, Olaf pace

    dt = 0.01  # 100 Hz
    print("Starting Olaf walk... Ctrl+C to stop")

    try:
        while True:
            t_start = time.monotonic()

            # ── Read joint states ──
            joint_pos = np.zeros(12)
            joint_vel = np.zeros(12)
            for i, (bus_type, motor_id) in enumerate(JOINT_ORDER):
                if bus_type == "rs":
                    name = f"rs_{motor_id}"
                    pos, vel, _, _ = rs_bus.read_operation_frame(name)
                    joint_pos[i] = pos
                    joint_vel[i] = vel
                else:
                    m = dm_motors[motor_id]
                    dm_ctrl.poll_feedback()
                    joint_pos[i] = m.state.get("pos", 0.0)
                    joint_vel[i] = m.state.get("vel", 0.0)

            # ── Build observation ──
            roll, pitch, yaw, gx, gy, gz = read_imu()
            obs = np.concatenate([
                joint_pos, joint_vel,
                [roll, pitch],
                [gx, gy, gz],
                command_velocity,
                prev_action,
            ]).astype(np.float32)
            obs = obs_filter(obs)

            # ── Run policy ──
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0))
                action = action.squeeze(0).numpy()

            action = np.clip(action, -1.0, 1.0)
            targets = DEFAULT_POSE + action * ACTION_SCALE
            prev_action = action.copy()

            # ── Send commands ──
            for i, (bus_type, motor_id) in enumerate(JOINT_ORDER):
                if bus_type == "rs":
                    name = f"rs_{motor_id}"
                    rs_bus.write_operation_frame(
                        name, position=targets[i],
                        kp=float(KP[i]), kd=float(KD[i])
                    )
                else:
                    m = dm_motors[motor_id]
                    m.send_cmd_mit(
                        target_position=float(targets[i]),
                        target_velocity=0.0,
                        stiffness=float(KP[i]),
                        damping=float(KD[i]),
                        feedforward_torque=0.0,
                    )

            # ── Maintain loop rate ──
            elapsed = time.monotonic() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dm_ctrl.disable_all()
        rs_bus.disconnect()
        print("All actuators disabled.")


if __name__ == "__main__":
    main()
```

### 9.2 Deployment Checklist

1. **Home all actuators first**: `python all_homing.py`
2. **Start in stand mode**: Run the Phase 1 (standing) policy for 10 seconds to verify stability
3. **Transition to walk**: Gradually ramp `command_velocity[0]` from 0 to 0.3 m/s over 3 seconds
4. **Monitor temperatures**: `read_operation_frame()` returns temperature — log and alert above 60C
5. **Emergency stop**: Ctrl+C triggers `disconnect()` which disables all torque output

---

## 10. Safety

### Software Safeguards

```python
# Add to the control loop:

# 1. Torque limits (hardware already enforces, but double-check)
MAX_TORQUES = {
    "rs-02": 17.0, "rs-03": 60.0, "4340P": 28.0, "4310": 10.0
}

# 2. Position limits — never command beyond safe range
JOINT_LIMITS = np.array([
    1.0, 0.5, 1.5, 2.5, 0.8, 0.5,   # Left leg
    1.0, 0.5, 1.5, 2.5, 0.8, 0.5,   # Right leg
])
targets = np.clip(targets, -JOINT_LIMITS, JOINT_LIMITS)

# 3. Velocity limits — reject sudden jumps
MAX_DELTA = 0.1  # rad per step
delta = targets - prev_targets
targets = prev_targets + np.clip(delta, -MAX_DELTA, MAX_DELTA)

# 4. Watchdog — if no IMU data for 100ms, go limp
if time.monotonic() - last_imu_time > 0.1:
    print("IMU timeout — disabling torque!")
    break
```

### Physical Safeguards

- **Gantry or safety harness** for initial real-world tests
- **Foam padding** on the floor around the robot
- **Kill switch** — hardware e-stop that cuts motor power supply
- **Current-limited power supply** during development
- **Never test alone** — have someone at the e-stop

---

## 11. References

### Frameworks & Tools

| Resource | URL | Notes |
|----------|-----|-------|
| MuJoCo | mujoco.org | Physics simulation |
| Isaac Gym | NVIDIA-Omniverse/IsaacGymEnvs | GPU-parallel training |
| legged_gym | leggedrobotics/legged_gym | ETH RSL locomotion framework |
| RSL-RL | leggedrobotics/rsl_rl | PPO implementation for legged robots |
| stable-baselines3 | DLR-RM/stable-baselines3 | General RL algorithms |

### Key Papers

- **"Learning Agile and Dynamic Motor Skills for Legged Robots"** (Hwangbo et al., 2019) — actuator network for sim-to-real
- **"Sim-to-Real: Learning Agile Locomotion For Quadrupeds"** (Tan et al., 2018) — domain randomization strategy
- **"Learning to Walk in Minutes Using Massively Parallel Deep RL"** (Rudin et al., 2022) — Isaac Gym legged locomotion
- **"Robust and Versatile Bipedal Jumping Control through RL"** (Li et al., 2023) — bipedal RL with sim-to-real
- **"DreamWaQ: Learning Robust Locomotion Over Uneven Terrain"** (Agarwal et al., 2023) — terrain adaptation

### Style Reference

- Watch Olaf's walk in *Frozen* (2013) — particularly the "In Summer" sequence for the most expressive waddle cycles. The key motion features to capture: lateral sway, bouncy CoM, short quick steps, and slight toe-out.
