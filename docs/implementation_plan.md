# Implementation Plan

Milestone: 0
Date: 2026-07-21
Project root: `/home/lab_726/uav_rendezvous_rl`

## Confirmed Decisions

| Decision | Confirmed value |
| --- | --- |
| Initial action space | Short-horizon target velocity `v_cmd_w in R^3` |
| Actor output interpretation | Actor outputs raw action; environment maps with `v_cmd_w = v_max * tanh(a_raw)` |
| Direct motor/force control | Forbidden for Actor in V0 |
| M1-M4 offset vector | `b_des_w = [d_offset, 0, 0]` |
| M5+ offset vector | Per-episode sampled `b_des_w`, with `||b_des_w|| = d_offset` |
| Initial offset distance | `d_offset = 5.0 m` |
| Curriculum | `5.0 m -> 3.0 m -> 2.0 m -> 1.0 m` |
| Placeholder radii | `r_ego = 0.20 m`, `r_target = 0.20 m` |
| Safety margin | `safety_margin = 0.35 m` |
| Initial safety distance | `d_safe = 0.75 m` |
| M1-M5 asset route | Simplified `RigidObject`/sphere and simplified velocity dynamics |
| Pegasus/PX4/ROS 2 | Not used in M1-M5 V0 |
| First high-fidelity route | Introduce Crazyflie or Isaac Lab `Multirotor/Thruster` only after simplified V0 is stable |

## Coordinate And Variable Conventions

All project-level position, velocity, and offset quantities use the world-aligned frame `w` and must include `_w` in variable names.

For vectorized Isaac Lab environments, `w` means the per-environment local world-aligned frame whose origin is `scene.env_origins[env_id]`. When writing poses to Isaac Sim, env-local positions are converted to simulation world positions by adding `scene.env_origins`. Relative quantities are origin-invariant.

Fixed definitions:

```text
p_rel_w = p_target_w - p_ego_w
v_rel_w = v_target_w - v_ego_w
e_offset_w = p_ego_w - p_target_w - b_des_w
```

Initial target condition:

```text
p_ego_w = p_target_w + b_des_w
b_des_w = [d_offset, 0, 0]
d_offset = 5.0 m
```

Safety condition:

```text
||p_target_w - p_ego_w|| < d_safe
```

If true, the episode terminates immediately and logs `Episode_Termination/collision_risk`.

## Recommended Directory Structure

```text
/home/lab_726/uav_rendezvous_rl/
  README.md
  pyproject.toml
  docs/
    environment_audit.md
    implementation_plan.md
  scripts/
    train.py
    play.py
    zero_agent.py
    random_agent.py
  source/
    uav_rendezvous_rl/
      pyproject.toml
      setup.py
      config/
        extension.toml
      uav_rendezvous_rl/
        __init__.py
        tasks/
          __init__.py
          direct/
            __init__.py
            uav_rendezvous_env.py
            uav_rendezvous_env_cfg.py
            agents/
              __init__.py
              rsl_rl_ppo_cfg.py
        motions/
          __init__.py
          target_motion.py
        controllers/
          __init__.py
          baseline.py
        utils/
          __init__.py
          math.py
          logging.py
          randomization.py
  tests/
    test_relative_state.py
    test_target_motion.py
    test_rewards.py
    test_env_smoke.py
```

M1 should create only the minimal subset required for editable install, Gymnasium task registration, reset/step, scripts, and tests. More advanced packages can be added in later milestones only when needed.

## Simulation And Control Timing

Initial timing for M1-M5 simplified dynamics:

| Quantity | Value | Notes |
| --- | --- | --- |
| Physics time step | `sim.dt = 0.01 s` | 100 Hz simulation |
| DirectRLEnv decimation | `2` | two physics substeps per policy step |
| Policy step | `step_dt = 0.02 s` | 50 Hz policy/action update |
| Low-level velocity tracking | 100 Hz | updated each physics substep in `_apply_action()` |
| Target motion integration | 100 Hz | update target kinematics every physics substep |
| Episode length M1 smoke | 10-20 s | M1 may use short timeouts for test speed |
| Episode length M2+ | 20 s | long enough for rendezvous behavior |
| Baseline prediction horizon | `T_pred = 0.5 s` | used in M4 |
| PPO rollout length V0 | 64 steps per env | about 1.28 s at 50 Hz |
| Success hold time | 1.0 s | about 50 policy steps |

When switching to Crazyflie or `Multirotor/Thruster` after simplified V0 is stable, evaluate `sim.dt = 0.005 s` and `decimation = 4` to keep a 50 Hz policy while improving dynamics stability.

## Asset And Dynamics Route

### M1-M4

Use simplified placeholders:

| Entity | Recommended representation |
| --- | --- |
| `ego` | `RigidObject` sphere, radius `0.20 m` |
| `target` | `RigidObject` sphere, radius `0.20 m` |
| Target dynamics | Kinematic truth update from target motion generator |
| Ego dynamics M1-M2 | Fixed or simple velocity state update |
| Ego dynamics M4-M5 | Acceleration-limited velocity tracking |

The target may be represented as a kinematic rigid object or as a visual rigid object whose root pose and velocity are written explicitly. Contact with ego is not part of the task; proximity below `d_safe` terminates as collision risk.

### M5 V0

Train on simplified dynamics first:

```text
a_raw in R^3
v_cmd_w = v_max * tanh(a_raw)
a_track_w = clamp((v_cmd_w - v_ego_w) / tau_v, -a_max, a_max)
v_ego_w <- clamp(v_ego_w + a_track_w * dt, -v_abs_max, v_abs_max)
p_ego_w <- p_ego_w + v_ego_w * dt
```

The Actor outputs only `a_raw`. The environment owns action mapping, velocity/acceleration limiting, and low-level tracking.

### V1 After M5 Stability

Introduce one high-fidelity route:

| Option | Pros | Risks |
| --- | --- | --- |
| Isaac Lab `CRAZYFLIE_CFG` with direct thrust/moment | Already has Direct RL example | Requires attitude/thrust control tuning |
| Isaac Lab `Multirotor/Thruster` | More realistic per-thruster dynamics | More configuration and stability work |

Pegasus, PX4, and ROS 2 remain out of scope until the simplified policy and high-fidelity Isaac Lab route are verified.

## Action Space V0

Recommended and confirmed V0 action:

```text
a_raw in R^3
v_cmd_w = v_max * tanh(a_raw)
```

Initial limits:

| Parameter | Suggested value | Notes |
| --- | --- | --- |
| `v_max` | `3.0 m/s` | conservative initial training value |
| `v_abs_max` | `5.0 m/s` | hard state limit |
| `a_max` | `2.0 m/s^2` | simplified acceleration limit |
| `tau_v` | `0.25 s` | velocity tracking time constant |
| action clipping | `[-1, 1]` at wrapper plus `tanh` in env | protects against raw outliers |

Do not output B-spline control points in V0. Add trajectory-parameterized actions only after V0 velocity action trains reliably.

## Actor Observation Definition

The Actor receives deployable information only. It must not receive target future state, target future control, target full future trajectory, target motion mode label, trajectory generator parameters, or simulator-only internals.

V0 Actor observation proposal:

| Name | Dim | Frame and unit | Deployable |
| --- | ---: | --- | --- |
| `p_rel_w` | 3 | world-aligned, m | yes |
| `v_rel_w` | 3 | world-aligned, m/s | yes |
| `v_ego_w` | 3 | world-aligned, m/s | yes |
| `R_ego_6d` | 6 | continuous ego attitude representation | yes |
| `omega_ego_b` | 3 | ego body, rad/s | yes |
| `last_action` | 3 | previous raw or normalized action | yes |
| `b_des_w` | 3 | world-aligned, m | yes |
| `d_offset` | 1 | m | yes, redundant with `||b_des_w||` but useful for curriculum logging |

Total V0 actor observation dimension: `25`.

Requirement: `b_des_w` is always included from the beginning. `d_offset` alone is not sufficient and must not replace `b_des_w`.

For recurrent PPO, the GRU hidden state provides implicit target-motion trend estimation from historical observations. The Actor does not explicitly output or consume a future target trajectory.

## Critic Observation Definition

The Critic may use privileged current truth to improve training. These quantities are not available to the Actor and must not leak into the policy observation group.

V0 Critic observation should be returned under the `critic` observation group and configured with:

```python
obs_groups = {
    "policy": ["policy"],
    "critic": ["critic"],
}
```

### Critic Privileged Information List

| Name | Dim | Description |
| --- | ---: | --- |
| Actor observation | 25 | Same current deployable observation used by Actor |
| `p_ego_w` | 3 | ego true position, env-local world frame, m |
| `p_target_w` | 3 | target true position, env-local world frame, m |
| `v_target_w` | 3 | target true velocity, env-local world frame, m/s |
| `a_target_w` | 3 | target true current acceleration, env-local world frame, m/s^2 |
| `R_target_6d` | 6 | target true attitude representation |
| `omega_target_b` | 3 | target true angular velocity, body frame, rad/s |
| `target_motion_mode_one_hot` | 4 | ConstantVelocity, ConstantAcceleration, ConstantTurn, PiecewiseAcceleration |
| `target_motion_current_params` | 6 | current generator parameters, padded and normalized; no full future trajectory |
| `episode_phase` | 1 | normalized episode time in `[0, 1]` |

Total V0 critic observation dimension: `57`.

The critic may later receive exact dynamics randomization parameters, exact wind/disturbance values, mass/inertia, control delay state, and observation-noise state. Any privileged item must be explicitly added to this section before implementation.

## Target Motion Library

Milestone 3 introduces a common `TargetMotionGenerator` interface.

Required implementations:

| Generator | State update |
| --- | --- |
| `ConstantVelocity` | `p_target_w(t) = p0_w + v0_w * t` |
| `ConstantAcceleration` | `p_target_w(t) = p0_w + v0_w * t + 0.5 * a_w * t^2` |
| `ConstantTurn` | horizontal coordinated turn with fixed altitude and yaw rate |
| `PiecewiseAcceleration` | piecewise constant `a_target_w` over seeded segments |

Requirements:

| Requirement | Detail |
| --- | --- |
| Batch support | All tensors shape `(num_envs, ...)` |
| No Python per-env loop in step | Use PyTorch vectorization |
| Seeded reset | Same seed and env count must reproduce parameters |
| Split configs | Train, validation, and test parameter ranges are separate |
| Actor leakage | Actor never receives motion mode, generator parameters, or future schedule |
| Critic privilege | Critic may receive current mode and current parameters as listed above |

## Reward Function V0

All reward components must be logged separately.

Recommended reward terms:

| Log key | Formula sketch | Purpose |
| --- | --- | --- |
| `Episode_Reward/offset` | `exp(-||e_offset_w||^2 / sigma_offset^2)` | reach desired non-contact offset |
| `Episode_Reward/relative_velocity` | `-||v_ego_w - v_target_w||^2` | match target velocity near rendezvous |
| `Episode_Reward/progress` | `prev(||e_offset_w||) - ||e_offset_w||` | reward improvement |
| `Episode_Reward/action_smoothness` | `-||a_raw_t - a_raw_{t-1}||^2` | reduce twitching |
| `Episode_Reward/action_magnitude` | `-||a_raw||^2` | avoid saturated commands |
| `Episode_Reward/safety_distance` | barrier or penalty near `d_safe` | maintain non-contact separation |
| `Episode_Reward/speed_limit` | penalty if `||v_ego_w|| > v_abs_max` | keep dynamics feasible |
| `Episode_Reward/accel_limit` | penalty if acceleration limiter saturates | discourage infeasible tracking |
| `Episode_Reward/attitude_rate` | penalty for excessive `omega_ego_b` | avoid unstable attitude when high-fidelity model is introduced |
| `Episode_Reward/workspace` | penalty near/outside workspace bounds | keep environments bounded |
| `Episode_Reward/success_bonus` | bonus for held success | stabilize final behavior |

Initial success condition:

```text
||e_offset_w|| < 0.50 m
||v_ego_w - v_target_w|| < 0.30 m/s
||p_target_w - p_ego_w|| >= d_safe
condition held for success_hold_s = 1.0 s
```

Thresholds should be configuration fields and can be tightened during curriculum.

## Termination Conditions

Each termination reason must be counted separately.

| Log key | Condition |
| --- | --- |
| `Episode_Termination/time_out` | episode reaches `max_episode_length` |
| `Episode_Termination/collision_risk` | `||p_target_w - p_ego_w|| < d_safe` |
| `Episode_Termination/workspace_violation` | ego or target leaves configured bounds |
| `Episode_Termination/height_violation` | altitude outside `[z_min, z_max]` |
| `Episode_Termination/speed_violation` | speed exceeds hard maximum |
| `Episode_Termination/attitude_violation` | high-fidelity model exceeds roll/pitch limit |
| `Episode_Termination/nan_or_inf` | any critical tensor is non-finite |
| `Episode_Termination/target_motion_invalid` | target generator emits invalid state |

## Testing Plan

### Pure PyTorch Unit Tests

Run without launching Isaac Sim where possible.

| Test | Purpose |
| --- | --- |
| `test_relative_state.py` | Verify `p_rel_w`, `v_rel_w`, and `e_offset_w` definitions and signs |
| `test_target_motion.py` | Verify analytic target motion for constant velocity and acceleration |
| `test_rewards.py` | Verify reward terms are finite and have expected monotonicity |
| `test_randomization.py` | Verify seeded reset and per-env randomization reproducibility |

### Isaac Lab Smoke Tests

Run through `/home/lab_726/IsaacLab/isaaclab.sh` with Conda inactive and ROS `PYTHONPATH` cleared.

| Test | Purpose |
| --- | --- |
| editable install | package can be installed with `pip install -e` |
| Gym task registration | `gym.make(task_id, cfg=...)` works |
| 16-env reset/step | reset and one step succeed on cuda:0 |
| random 10000 steps | no NaN, Inf, CUDA error, or crash |
| zero 10000 steps | deterministic no-op/safe action path works |

## Milestones And Acceptance Criteria

### Milestone 1: Minimal External Project

Create only a minimal external Direct RL project.

Acceptance criteria:

| Requirement | Acceptance check |
| --- | --- |
| Editable install | `pip install -e source/uav_rendezvous_rl` succeeds using Isaac Lab launcher |
| Gym registration | task ID appears and `gym.make` succeeds |
| 16 envs | environment creates 16 parallel envs on `cuda:0` |
| Reset and step | `reset()` and random `step()` return finite tensors |
| 10000 steps | random actions complete without NaN, Inf, CUDA error, or crash |
| Tests | pytest suite passes |

M1 may use placeholder assets and minimal reward/done logic. It must not implement the full dual-UAV motion library or training.

### Milestone 2: Dual UAV Truth Environment

Acceptance criteria:

| Requirement | Acceptance check |
| --- | --- |
| Entities | each env has `ego` and `target` placeholders |
| Relative state | `p_rel_w` and `v_rel_w` tests pass |
| Target motion | constant velocity fixed-height motion matches analytic solution |
| Reset | both entities restore correct per-env states |
| Randomization | 16 envs independently randomized with fixed-seed reproducibility |
| Stability | 10000 steps complete without non-finite values or CUDA errors |

### Milestone 3: Target Motion Library

Acceptance criteria:

| Requirement | Acceptance check |
| --- | --- |
| Interface | common `TargetMotionGenerator` implemented |
| Generators | ConstantVelocity, ConstantAcceleration, ConstantTurn, PiecewiseAcceleration implemented |
| Batch | vectorized per-env parameters |
| Reproducibility | fixed seed reproduces sampled motion |
| Splits | train/validation/test ranges are separate |
| Actor isolation | Actor observation contains no mode labels or generator parameters |

### Milestone 4: Deterministic Non-Learning Baseline

Acceptance criteria:

| Requirement | Acceptance check |
| --- | --- |
| Prediction | `p_target_pred_w(T) = p_target_w + v_target_w * T` |
| Goal | `p_goal_w = p_target_pred_w + b_des_w` |
| Control | ego follows goal with velocity/acceleration limits |
| Performance | simple constant-velocity target can be approached to offset state |
| Safety | no `collision_risk` termination in nominal baseline test |
| Metrics | offset error, relative velocity, and success rate are logged |

If M4 baseline fails, do not enter reinforcement learning.

### Milestone 5: First RL Environment V0

Acceptance criteria:

| Requirement | Acceptance check |
| --- | --- |
| Action | `v_cmd_w` action mapping and low-level limiter implemented |
| Actor obs | includes `b_des_w` and no forbidden future/privileged info |
| Critic obs | privileged current truth under `critic` group |
| Rewards | each reward component logged separately |
| Terminations | each termination reason counted separately |
| Random action stability | 10000 steps stable before training |
| Short training | initial PPO sanity run improves or remains stable |

### Milestone 6: Recurrent PPO And Implicit Prediction

Acceptance criteria:

| Requirement | Acceptance check |
| --- | --- |
| GRU policy | `RslRlPpoActorCriticRecurrentCfg(rnn_type="gru")` used |
| Asymmetry | `obs_groups={"policy": ["policy"], "critic": ["critic"]}` |
| Done masks | per-env done clears GRU hidden state |
| Normalization | actor and critic normalization configured intentionally |
| Checkpoints | save/resume verified |
| Play/evaluate | deterministic inference works and resets recurrent state correctly |

### Milestone 7: Curriculum And Robustness

Acceptance criteria:

| Stage | Added difficulty |
| --- | --- |
| 1 | more complex target maneuvers |
| 2 | wider initial distance and bearing randomization |
| 3 | ego mass/dynamics randomization |
| 4 | wind and control delay |
| 5 | position/velocity observation noise |
| 6 | random observation delay |
| 7 | dropped observations |
| 8 | distance-dependent observation error |
| 9 | first-stage estimator output integration |

At every stage, Actor observations must remain deployable and must not include target future state, target future command, motion mode label, or generator parameters.

## M1 Planned Files

When M1 is approved, create the following minimal files:

```text
README.md
pyproject.toml
scripts/train.py
scripts/play.py
scripts/zero_agent.py
scripts/random_agent.py
source/uav_rendezvous_rl/pyproject.toml
source/uav_rendezvous_rl/setup.py
source/uav_rendezvous_rl/config/extension.toml
source/uav_rendezvous_rl/uav_rendezvous_rl/__init__.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/__init__.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/__init__.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env_cfg.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/__init__.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py
tests/test_env_registration.py
tests/test_env_smoke.py
```

M1 scripts should support finite `--steps` so smoke tests do not rely on manually closing Isaac Sim.

## M1 Planned Commands

All commands must run with Conda inactive and ROS `PYTHONPATH` cleared.

Preflight:

```bash
test -z "${CONDA_PREFIX:-}"
test -z "${CONDA_DEFAULT_ENV:-}"
```

Editable install:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pip install -e /home/lab_726/uav_rendezvous_rl/source/uav_rendezvous_rl
```

Pytest:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pytest /home/lab_726/uav_rendezvous_rl/tests -q
```

Zero-agent smoke run:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/zero_agent.py --task Isaac-Uav-Rendezvous-Direct-v0 --num_envs 16 --device cuda:0 --headless --steps 10000
```

Random-agent smoke run:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/random_agent.py --task Isaac-Uav-Rendezvous-Direct-v0 --num_envs 16 --device cuda:0 --headless --steps 10000
```

M1 completion will stop before any training. A Git repository can be initialized and committed only after explicit user confirmation.
