# OLAF SDK Deployment Guide (rsl_rl → on-board computer)

Audience: an AI/engineer fine-tuning the on-robot SDK so it consumes the
`policy.onnx` produced by `scripts/rsl_rl/export_policy.py` from this repo.
Treat this file as the single source of truth for: joint order, observation
layout, action mapping, control rate, PD gains, and IMU semantics. Every
constant below is cross-referenced to the file/line that defines it in
training so the two stay in sync.

---

## 1. Pipeline at a glance

```
    Isaac Lab                                         Robot computer
  ─────────────                                      ────────────────
  Olaf-Velocity-Flat-v0  ──train──►  rsl_rl PPO  ──►  policy.pt
        (50 Hz, 12 joints)                       └──►  policy.onnx ◄── SDK
                                                                       │
                                                       50 Hz   ┌───────┴────────┐
                                                               │ build obs (45) │
                                                               │ run ONNX       │
                                                               │ q_target = q0  │
                                                               │   + 0.5·action │
                                                               └────────────────┘
                                                                       │
                                                              200 Hz PD inner loop
                                                                       │
                                                                  joint torques
```

Key invariant: the policy was trained at **50 Hz** (sim.dt 0.005 s,
decimation 4). The PD that closes around it ran at **200 Hz** in sim. The SDK
must reproduce this timing — see §6.

---

## 2. Joint order (CRITICAL — get this wrong and the robot falls over)

Isaac Lab interleaves joints `[L, R]` per family (NOT all-L then all-R, NOT
all-leg-1 then all-leg-2). This is the order printed by
`scripts/zero_agent.py` from `robot.data.joint_names`:

| idx | joint name              | actuator family | kp     | kd    | τ_max (N·m) |
|----:|-------------------------|-----------------|-------:|------:|------------:|
|   0 | `l_hip_yaw_joint`       | RS02            | 40.000 | 3.000 |        11.9 |
|   1 | `r_hip_yaw_joint`       | RS02            | 40.000 | 3.000 |        11.9 |
|   2 | `l_hip_roll_joint`      | RS03            | 78.957 | 5.027 |        42.0 |
|   3 | `r_hip_roll_joint`      | RS03            | 78.957 | 5.027 |        42.0 |
|   4 | `l_hip_pitch_joint`     | RS03            | 78.957 | 5.027 |        42.0 |
|   5 | `r_hip_pitch_joint`     | RS03            | 78.957 | 5.027 |        42.0 |
|   6 | `l_knee_pitch_joint`    | RS03            | 78.957 | 5.027 |        42.0 |
|   7 | `r_knee_pitch_joint`    | RS03            | 78.957 | 5.027 |        42.0 |
|   8 | `l_ankle_pitch_joint`   | RS00            | 16.581 | 1.056 |        14.0 |
|   9 | `r_ankle_pitch_joint`   | RS00            | 16.581 | 1.056 |        14.0 |
|  10 | `l_ankle_roll_joint`    | RS00            | 16.581 | 1.056 |        14.0 |
|  11 | `r_ankle_roll_joint`    | RS00            | 16.581 | 1.056 |        14.0 |

Source: `source/olaf_bipedal_robot/robots/olaf.py:70-118` (gains, effort
limits) + this is exactly the order the user printed during a zero_agent run:
`['l_hip_yaw_joint', 'r_hip_yaw_joint', 'l_hip_roll_joint', 'r_hip_roll_joint',
'l_hip_pitch_joint', 'r_hip_pitch_joint', 'l_knee_pitch_joint',
'r_knee_pitch_joint', 'l_ankle_pitch_joint', 'r_ankle_pitch_joint',
'l_ankle_roll_joint', 'r_ankle_roll_joint']`.

The SDK must keep one explicit `JOINT_ORDER` constant in this exact order and
build a permutation from it to the on-robot bus IDs. **Do not assume the
firmware bus order matches.** Verify on-bench by commanding each index one at
a time and confirming the right joint moves.

---

## 3. Default joint pose (`q0`)

This is the bent-knee stance the policy was trained around. Action 0
corresponds to this pose.

| idx | joint              | q0 (rad) |
|----:|--------------------|---------:|
|   0 | l_hip_yaw_joint    |    0.000 |
|   1 | r_hip_yaw_joint    |    0.000 |
|   2 | l_hip_roll_joint   |    0.000 |
|   3 | r_hip_roll_joint   |    0.000 |
|   4 | l_hip_pitch_joint  |    0.900 |
|   5 | r_hip_pitch_joint  |    0.900 |
|   6 | l_knee_pitch_joint |    1.650 |
|   7 | r_knee_pitch_joint |    1.650 |
|   8 | l_ankle_pitch_joint|    0.700 |
|   9 | r_ankle_pitch_joint|    0.700 |
|  10 | l_ankle_roll_joint |    0.000 |
|  11 | r_ankle_roll_joint |    0.000 |

Source: `_BENT_KNEE_INIT` in `source/olaf_bipedal_robot/robots/olaf.py:31-38`.

L and R use the **same numeric values** because the URDF mirroring is encoded
in the joint axes (verify: `axis="0 -1 0"` for `l_hip_pitch_joint` vs
`axis="0 1 0"` for `r_hip_pitch_joint` in
`source/assets/MJCF/olaf_robstride.xml:94,130`). Sign-flipping L/R targets in
the SDK will mirror the gait incorrectly.

Steady-state characteristics under zero action (measured via `zero_agent`):

- `base_link` height above ground: **~0.301 m** (NOT the 0.45 m spawn height).
- `*_ankle_roll_link` z above ground: **~0.0475 m** (foot link origin).

The SDK should ramp the joints from the as-found pose to `q0` at boot before
enabling the policy. A safe ramp is ~3 s linear interpolation while the
torso is supported.

---

## 4. Observation vector (length = 45)

Built every policy tick (50 Hz) and fed to the ONNX as `obs[1, 45]`. Order
and per-term scaling MUST match exactly — the policy was trained on these
shapes.

```
idx range │ term                        │ scale │ shape │ source
──────────┼─────────────────────────────┼───────┼───────┼─────────────────────────
   0:3    │ imu_ang_vel                 │ 0.20  │ (3,)  │ olaf_env_cfg.py:126
   3:6    │ imu_projected_gravity       │ 1.00  │ (3,)  │ olaf_env_cfg.py:127-130
   6:18   │ joint_pos_rel = q − q0      │ 1.00  │ (12,) │ olaf_env_cfg.py:131
  18:30   │ joint_vel_rel × 0.05        │ 0.05  │ (12,) │ olaf_env_cfg.py:132
  30:42   │ last_action (raw, pre-scale)│ 1.00  │ (12,) │ olaf_env_cfg.py:133
  42:45   │ velocity_command (vx,vy,wz) │ 1.00  │ (3,)  │ olaf_env_cfg.py:134
──────────┴─────────────────────────────┴───────┴───────┴─────────────────────────
```

Definitions are pinned to the underlying Isaac Lab functions:

- `imu_ang_vel`     = `Imu.data.ang_vel_b` — angular velocity in the IMU
                       body frame, rad/s
                       (`isaaclab/envs/mdp/observations.py:346`).
- `imu_projected_gravity` = `Imu.data.projected_gravity_b` — the world gravity
                       direction expressed in IMU frame, **already
                       normalized** (≈ `[0, 0, -1]` when level)
                       (`isaaclab/envs/mdp/observations.py:331`).
                       This is NOT raw accelerometer; it is gravity only.
- `joint_pos_rel`   = `q − q_default` per joint, rad
                       (`isaaclab/envs/mdp/observations.py:212`).
- `joint_vel_rel`   = `qd − qd_default` per joint; `qd_default = 0`, so this
                       is just `qd`, rad/s
                       (`isaaclab/envs/mdp/observations.py:257`).
- `last_action`     = the **raw action vector** the policy returned on the
                       previous tick, BEFORE the 0.5 scale and BEFORE adding
                       q0 (`isaaclab/envs/mdp/observations.py:657`). On the
                       very first tick this is zeros.
- `velocity_command`= `(vx, vy, wz)` in m/s, m/s, rad/s.

### IMU mounting

The training env mounts the IMU on `base_link` with zero offset and identity
rotation (`olaf_env_cfg.py:412-419`). The URDF `imu_link` is merged into
`base_link` by `merge_fixed_joints=True`, so on the robot the IMU body frame
== `base_link` frame. The SDK must report `ang_vel` and `projected_gravity`
**in this frame**, not the raw chip frame. Apply the chip-to-base extrinsic
once at boot.

### Computing `imu_projected_gravity` from a real IMU

```python
# `g_world` is the unit gravity vector expressed in world frame, ≈ [0, 0, -1]
# `R_world_from_base` is the orientation estimate from your AHRS (quaternion)
g_proj_base = R_world_from_base.T @ g_world          # base-frame gravity unit vec
```

Sanity check at standstill on flat ground: `g_proj_base ≈ [0, 0, -1]`.

### Velocity command source

This is the user/teleop input, not a measurement. Hold (0, 0, 0) when no
command is active — the policy was trained to **march in place** at zero
command (see §10).

---

## 5. Action vector (length = 12)

```
q_target[i] = q0[i] + 0.5 * action[i]      i = 0 .. 11
```

- `0.5` is `JointPositionActionCfg.scale`
  (`olaf_env_cfg.py:78`, also reasserted at line 422).
- Order matches `JOINT_ORDER` in §2.
- No squashing or clipping of `action` happens during training
  (`PPORunnerCfg.algorithm.clip_actions` is unset → `clip_actions=None` in
  `RslRlVecEnvWrapper`). The actor outputs a raw Gaussian mean. In practice
  values stay within ~±5 because the policy converges to small deltas, but
  the SDK MUST clamp **`q_target`** to the URDF limits (§7), not pre-clamp
  the action.

The exported ONNX is deterministic (mean action only — see
`scripts/rsl_rl/export_policy.py:96` `_MeanActionWrapper`). No sampling,
no temperature.

---

## 6. Control rate and PD inner loop

| quantity            | value            | source                            |
|---------------------|------------------|-----------------------------------|
| `sim.dt`            | 0.005 s          | base velocity_env_cfg.py:311      |
| `decimation`        | 4                | base velocity_env_cfg.py:308      |
| Policy rate         | 50 Hz (0.020 s)  | sim.dt × decimation               |
| PD update rate      | 200 Hz (0.005 s) | runs every sim sub-step           |
| Episode length (sim)| 20 s             | base velocity_env_cfg.py:309      |

Pseudo-code for the SDK control loop:

```python
last_action = np.zeros(12, dtype=np.float32)

while running:                             # 50 Hz scheduler
    obs = build_obs(imu, q, qd, last_action, cmd)   # §4
    action = onnx.run({"obs": obs[None]})[0][0]      # (12,)
    last_action = action                             # remember raw
    q_target = q0 + 0.5 * action                    # §5

    # --- inner PD at 200 Hz ---
    for _ in range(4):                               # decimation
        tau = KP * (q_target - q) - KD * qd          # §2 gains
        tau = np.clip(tau, -tau_limit, tau_limit)    # §2 ceiling
        write_torque(tau)
        sleep_to_200hz_tick()
        q, qd, imu = read_state()
```

Why the inner PD must run at 200 Hz, not 50 Hz: ankle_roll has very low
reflected inertia and a 50 Hz zero-order hold on velocity feedback acts as
effective negative damping. This was reproduced in `mujoco_play.py` —
recomputing `tau` per sub-step matches the Isaac Lab `ImplicitActuator`
behavior the policy trained against. See the comment at
`scripts/sim2sim/mujoco_play.py:207-213`.

---

## 7. Joint limits (clamp q_target here, not action)

From `source/assets/MJCF/olaf_robstride.xml:84-145` (which mirrors the URDF
exactly):

| idx | joint              | lower (rad) | upper (rad) |
|----:|--------------------|------------:|------------:|
|   0 | l_hip_yaw          |       −0.26 |        0.26 |
|   1 | r_hip_yaw          |       −0.26 |        0.26 |
|   2 | l_hip_roll         |     −0.2618 |       0.349 |
|   3 | r_hip_roll         |     −0.2618 |       0.349 |
|   4 | l_hip_pitch        |       −0.10 |        1.40 |
|   5 | r_hip_pitch        |       −0.10 |        1.40 |
|   6 | l_knee_pitch       |       −0.40 |        2.00 |
|   7 | r_knee_pitch       |       −0.40 |        2.00 |
|   8 | l_ankle_pitch      |       −0.65 |        1.05 |
|   9 | r_ankle_pitch      |       −0.65 |        1.05 |
|  10 | l_ankle_roll       |       −0.20 |        0.26 |
|  11 | r_ankle_roll       |       −0.20 |        0.26 |

**Note:** CLAUDE.md lists slightly different L/R asymmetric numbers for
hip_pitch and knee_pitch (e.g. `1.20 (L) / 1.40 (R)` for hip_pitch upper).
The URDF/MJCF that was actually used for training has the **symmetric**
limits in the table above. Trust the MJCF column — that's what the policy
saw.

`soft_joint_pos_limit_factor = 0.95` was applied to these in training
(`olaf.py:120`). For deployment, clamp `q_target` to the **hard** URDF
limits; let the natural policy behavior stay inside the soft band.

---

## 8. ONNX I/O contract

```
inputs  : {"obs":    float32, shape (1, 45)}    # batch dim is dynamic
outputs : {"action": float32, shape (1, 12)}
```

The exporter (`isaaclab_rl/rsl_rl/exporter.py:166`) wraps the actor so the
graph is `actor(normalizer(obs))`. The normalizer is part of the ONNX —
**the SDK must NOT apply any extra normalization, mean/var subtraction, or
clipping to obs**. Just feed the raw vector built per §4.

In this project `empirical_normalization = False` and `obs_normalization =
False` (`agents/rsl_rl_ppo_cfg.py:9,17`), so the baked-in normalizer is the
identity, but the SDK should not depend on that — always feed raw obs.

The skrl exporter at `scripts/skrl/export_policy.py` is a different code
path (RL framework not in use for this deployment). It writes a separate
`preprocessor.json` that the SDK would have to apply. **Use the rsl_rl
export only** — the user is deploying rsl_rl results.

Quick-check after loading the model:

```python
sess = onnxruntime.InferenceSession("policy.onnx")
zero_in = np.zeros((1, 45), dtype=np.float32)
out = sess.run(["action"], {"obs": zero_in})[0]
assert out.shape == (1, 12)
```

The output for an all-zero input is the policy's "rest" action, which
should be small (|action| < ~1 per joint). If it's wildly large the obs
layout is probably wrong.

---

## 9. Boot, watchdog, fault behavior

These aren't enforced by training — they're SDK responsibilities.

1. **Boot ramp.** Read current joint positions, linearly interpolate to `q0`
   over ~3 s with low gains (e.g. KP×0.3) before handing control to the
   policy.
2. **Latched zeros.** On the first policy tick, `last_action = zeros(12)`.
   Don't read it from anywhere else.
3. **Frame skip.** If the policy thread misses its 50 Hz deadline, hold the
   previous `q_target` rather than re-running PD with stale state. Repeated
   misses (>3 in a row) → cut torque.
4. **IMU loss / NaN gravity.** If `‖projected_gravity‖` deviates from 1 by
   more than ~0.1, or any obs entry is NaN, freeze `q_target` at `q0` and
   sound a fault.
5. **Tilt cutoff.** Trip when `g_proj_base[2] > -0.5` (torso pitched more
   than ~60° from upright). Training terminates an episode on `base_link` /
   `hip_*_link` / `knee_pitch_link` ground contact (`olaf_env_cfg.py:374-383`)
   so out-of-distribution body orientations have not been visited.
6. **Joint limit guard.** Clamp `q_target` to the §7 ranges. The PD loop's
   torque clip is a separate, second line of defense.

---

## 10. Behaviors recently changed (ACTIVE — affects deployment expectations)

These are the diffs visible against `origin/main` and they change how the
policy behaves on the robot. Make sure SDK QA tests cover them.

1. **March-in-place at zero command.** Three rewards lost their command-magnitude
   gating in `mdp/rewards.py`:
   - `bipedal_air_time_reward`  — air-time reward now applied even at `cmd≈0`.
   - `GaitReward`               — sync/async gait reward no longer gated on `cmd`.
   - `joint_position_penalty`   — `stand_still_scale` removed; deviation
                                  penalty is uniform.

   Combined with `rel_standing_envs=0.15` in
   `olaf_env_cfg.py:CommandsCfg.base_velocity` (15% of envs reset with exact
   `(0,0,0)`), the policy is trained to step in place when the command is
   zero rather than freezing into stance. **The SDK should not assume "zero
   command ⇒ no foot motion".** Pads/joints will keep moving slightly even at
   rest. This is by design.

2. **Stand-still demo command added** to `scripts/rsl_rl/play.py`
   (`(+0.0, +0.0, 0.0)` in `DEMO_PALETTE`). Useful to spot-check
   march-in-place behavior side-by-side with locomotion in the viewer.

3. **`scripts/rsl_rl/export_policy.py`** is the canonical export path. It
   handles both legacy rsl_rl (≤4.x) and rsl_rl ≥5.0 layouts — falling back
   to a manual `_MeanActionWrapper` export if the upstream exporter rejects
   the new MLPModel structure. Always run it on the **training machine**, not
   on the robot. Output dir defaults to `<checkpoint_dir>/exported/`.

4. **Foot clearance target = 0.10 m world z** (≈ 0.0525 m above the standing
   foot height of 0.0475 m); this informs how high the swing foot will lift.
   The robot's feet should clear obstacles up to ~5 cm reliably; anything
   taller is out-of-distribution.

5. **Base height target = 0.30 m.** The policy will try to keep `base_link`
   at this height. Mounting platforms / cables that pull the torso below
   ~0.27 m or above ~0.33 m will fight the policy.

---

## 11. Quick deployment checklist

- [ ] `policy.onnx` produced by `scripts/rsl_rl/export_policy.py` (NOT the
      skrl exporter).
- [ ] `JOINT_ORDER` constant in the SDK matches §2 verbatim, with a permutation
      table to firmware bus IDs verified by single-joint commands on bench.
- [ ] `q0` constant matches §3, applied as a 3 s boot ramp.
- [ ] Obs builder produces a 45-element float32 in the order in §4. Print
      the obs once on boot and diff against a screenshot from `mujoco_play.py`
      with the same `(vx, vy, wz)`.
- [ ] Inner PD runs at 200 Hz with the per-joint `KP/KD/τ_max` from §2.
- [ ] Action mapping is `q0 + 0.5 * action` (§5). `q_target` clamped to
      §7 limits.
- [ ] IMU reports angular velocity AND projected-gravity unit vector in
      `base_link` frame (§4 sub-section "IMU mounting").
- [ ] Watchdog (§9) wired up before the first untethered run.
- [ ] At idle (`cmd = 0,0,0`) the robot **marches in place** — that's
      expected behavior, not a control bug (§10).

---

## 12. File pointers (training side, for cross-referencing)

```
source/olaf_bipedal_robot/robots/olaf.py
    OLAF_CFG, _BENT_KNEE_INIT, per-joint kp/kd/τ/v limits.

source/olaf_bipedal_robot/tasks/manager_based/olaf_locomotion/olaf_env_cfg.py
    Action scale, observation layout, command ranges, IMU offset, terminations.

source/olaf_bipedal_robot/tasks/manager_based/olaf_locomotion/mdp/rewards.py
    Air-time / gait / joint-pos rewards (cmd-gating removed → march in place).

source/olaf_bipedal_robot/tasks/manager_based/olaf_locomotion/agents/rsl_rl_ppo_cfg.py
    Actor MLP shape (512,256,128, ELU), normalization off, PPO hyperparams.

scripts/rsl_rl/export_policy.py        — produces policy.onnx + policy.pt.
scripts/rsl_rl/play.py                 — Isaac Lab-side reference inference.
scripts/sim2sim/mujoco_play.py         — minimal end-to-end SDK template (MuJoCo).
source/assets/MJCF/olaf_robstride.xml  — joint limits + actuator setup the SDK should mirror.
```