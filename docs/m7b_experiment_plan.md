# M7B Experiment Plan

Date: 2026-07-24

Status: revised plan pending independent final review. Code implementation has not started.

## M7B Terminology

M7B uses three orthogonal numbering systems:

1. Implementation Blocks

- M7B-I1: pure PyTorch infrastructure
- M7B-I2: environment integration
- M7B-I3: training/play/evaluation contracts
- M7B-I4: tests and runtime audits

2. Qualification Gates

- M7B-G1: syntax and pure unit tests
- M7B-G2: nominal dynamics equivalence
- M7B-G3: environment and observation contracts
- M7B-G4: runtime and regression audits
- M7B-G5: clean startup and validation

3. Experiment Stages

- M7B-S0: clean nominal regression
- M7B-S1: dynamics randomization only
- M7B-S2: action delay only
- M7B-S3: wind and gust only
- M7B-S4: combined M7B robustness

Definitions:

- Implementation Block describes implementation order.
- Qualification Gate describes the entry condition for continuing work.
- Experiment Stage describes a training or evaluation scenario.
- The three numbering systems must not be substituted for each other.
- The retired Phase-5 letter labels are no longer used; use M7B-I1, M7B-I2, M7B-I3, and M7B-I4 instead.
- M7C is not authorized and is unrelated to the M7B numbering above.
- `docs/implementation_plan.md` old project-level Milestone 7 stages 1-9 are historical roadmap items; M7B-S0 through M7B-S4 are current sub-milestone experiment identifiers.

## Scope

M7B validates robustness of the non-contact offset rendezvous policy under:

1. Simplified ego dynamics parameter randomization.
2. Control execution delay through a per-environment action FIFO.
3. Steady wind and piecewise gust equivalent acceleration.
4. Combined training and validation of all factors.

The primary policy is feedforward PPO. GRU is a secondary ablation only. M7B does not add M7A observation degradation; the first M7B round uses the clean M7A observation pipeline with delay `0`, dropout `0`, noise `0`, and position/velocity updates at policy frequency.

M7B remains a non-contact offset rendezvous task. It must not reward contact, impact, target crossing, or collision risk. Existing `b_des_w`, `d_safe`, collision-risk termination, reward terms, success thresholds, workspace limits, and height limits remain unchanged unless a later user instruction explicitly authorizes a change.

M7C remains not authorized.

## Tasks

M7B adds two independent tasks:

```text
Isaac-Uav-Rendezvous-M7B-Feedforward-v0
Isaac-Uav-Rendezvous-M7B-GRU-v0
```

`Isaac-Uav-Rendezvous-M7B-Feedforward-v0` is the primary training and deployment candidate. `Isaac-Uav-Rendezvous-M7B-GRU-v0` is only a secondary M7B-S4 ablation after the feedforward M7B-S4 candidate is stable.

The accepted tasks below must not change behavior:

```text
Isaac-Uav-Rendezvous-Direct-v0
Isaac-Uav-Rendezvous-Baseline-v0
Isaac-Uav-Rendezvous-RL-v0
Isaac-Uav-Rendezvous-Recurrent-v0
Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0
Isaac-Uav-Rendezvous-M7A-GRU-v0
Isaac-Uav-Rendezvous-M7A-Feedforward-v0
```

## Formal Dynamics Definition

M7B extends the accepted M5-M7A translational ego dynamics without entering six-degree-of-freedom multirotor dynamics. The following equations are the only formal M7B dynamics definition.

```text
v_cmd_delayed_w = v_max * executed_squashed_action

tau_velocity = tau_v_nominal * tau_velocity_scale

acceleration_limit = a_max_nominal * acceleration_limit_scale

physical_speed_limit = v_abs_max_nominal * speed_limit_scale

a_track_raw_w = (v_cmd_delayed_w - v_ego_w) / tau_velocity

a_track_w = clamp_vector_norm(a_track_raw_w, acceleration_limit)

a_unclipped_w = a_track_w - linear_drag * v_ego_w + wind_acceleration_w

a_applied_w = clamp_vector_norm(a_unclipped_w, a_applied_abs_max)

v_candidate_w = v_ego_w + a_applied_w * physics_dt

v_ego_next_w = clamp_vector_norm(v_candidate_w, physical_speed_limit)

p_ego_next_w = p_ego_w + v_ego_w * physics_dt + 0.5 * a_applied_w * physics_dt**2
```

Notes:

- `v_max` is the existing Actor action mapping limit, nominally `3.0 m/s`.
- `tau_v_nominal = 0.25 s`.
- `a_max_nominal = 2.0 m/s^2`.
- `v_abs_max_nominal = 5.0 m/s`.
- `physics_dt` is the existing simulation physics timestep.
- All vector clamps are Euclidean norm clamps using `clamp_vector_norm` semantics.

M7B-S0 must keep the M5-M7A position integration formula. M7B must not use `p_ego_next_w = p_ego_w + v_ego_next_w * physics_dt` for the formal dynamics.

If the speed clamp triggers, M7B does not retroactively recompute or project this step's position. This matches the existing accepted behavior: position is integrated from the start-of-substep velocity and applied acceleration, then the velocity state is clamped for the next substep. Speed saturation is logged separately.

Tracking acceleration saturation and total applied acceleration safety saturation are separate events:

- Tracking saturation: `a_track_raw_w` is clamped by `acceleration_limit`.
- Total safety saturation: `a_unclipped_w` is clamped by `a_applied_abs_max`.

The existing reward `accel_limit` continues to reflect tracking saturation. Total safety saturation is an audit/safety counter and must not silently change reward semantics.

## Total Acceleration Safety Cap

M7B uses a fixed total applied acceleration safety cap:

```text
a_applied_abs_max = 4.0 m/s^2
```

This is equivalent to:

```text
2.0 * a_max_nominal
```

Requirements:

- Fixed config field, not domain randomized.
- Applied after drag and wind are added.
- Separate from `acceleration_limit_scale`.
- Config validation requires `a_applied_abs_max > 0`.
- Runtime audit reports trigger count and fraction.

Rationale for `4.0 m/s^2`:

```text
max tracking acceleration: 2.0 * 1.20 = 2.4 m/s^2
max steady wind: 2.0 * 0.20 = 0.4 m/s^2
max gust: 2.0 * 0.15 = 0.3 m/s^2
max drag at physical speed edge: 0.15 * (5.0 * 1.10) = 0.825 m/s^2
scalar upper bound: 2.4 + 0.4 + 0.3 + 0.825 = 3.925 m/s^2
```

The `4.0 m/s^2` cap covers the intended edge range while preventing unbounded total acceleration. A smaller value such as `3.6 m/s^2` would clip valid edge combinations too often.

## Action Mapping and Speed Limits

M7B keeps the existing Actor action mapping unchanged:

```text
squashed_action = tanh(raw_action)
v_cmd_w = v_max * squashed_action
v_max = 3.0 m/s
```

`speed_limit_scale` does not change `v_max`. It only affects the physical speed clamp:

```text
physical_speed_limit = v_abs_max_nominal * speed_limit_scale
```

The Actor does not receive `speed_limit_scale` or `physical_speed_limit`.

Action command range and physical velocity bound are independent concepts. A command may be harder to realize under high drag, longer delay, lower acceleration authority, or lower physical speed limit. This is an intended robustness condition, not a reason to change reward or Actor observations. The Critic receives `speed_limit_scale` as current-time privileged training information.

## Dynamics Randomization Parameters

All per-episode values are fixed for an environment until that environment resets. Wind/gust is the only time-varying disturbance process.

| Parameter | M7B-S0 | Formal randomized range | Meaning |
| --- | ---: | --- | --- |
| `tau_velocity_scale` | `1.0` | `[0.75, 1.50]` | Multiplier on nominal `tau_v` |
| `acceleration_limit_scale` | `1.0` | `[0.80, 1.20]` | Multiplier on nominal `a_max` |
| `speed_limit_scale` | `1.0` | `[0.90, 1.10]` | Multiplier on nominal physical `v_abs_max` |
| `linear_drag` | `0.0` | `[0.00, 0.15] 1/s` | Linear drag coefficient |
| `action_delay_steps` | `0` | integer `[0, 3]` | Policy-step execution delay |
| `steady_wind_magnitude` | `0.0` | `[0.00, 0.20] * a_max_nominal` | Steady acceleration magnitude |
| `gust_magnitude` | `0.0` | `[0.00, 0.15] * a_max_nominal` | Gust target acceleration magnitude |
| `gust_duration_steps` | disabled | integer `[10, 50]` | Policy steps per gust target |
| `a_applied_abs_max` | `4.0` | fixed `4.0 m/s^2` | Total safety cap, not randomized |

All ranges are config fields. If a future nominal dynamics value changes, these ranges must be reviewed before training.

## Action FIFO

M7B implements a per-environment FIFO storing squashed actions, not velocity commands.

```text
buffer shape = (num_envs, max_delay_steps + 1, 3)
L = max_delay_steps + 1
```

At every policy step `t`:

```text
1. s_t = tanh(raw_action_t)
2. buffer[:, write_index, :] = s_t
3. read_index[env] = (write_index - action_delay_steps[env]) mod L
4. executed_squashed_action[env] = buffer[env, read_index[env], :]
5. v_cmd_delayed_w = v_max * executed_squashed_action
6. write_index = (write_index + 1) mod L
```

This is push-then-read. It is required for `delay=0` to execute the current policy action immediately.

Delay semantics:

- `delay=0`: execute current policy-step action.
- `delay=1`: execute the previous policy-step action.
- `delay=N`: execute the action from `N` policy steps ago.

Reset semantics:

- Full reset and partial reset fill all FIFO slots for selected envs with zero squashed actions.
- Partial reset does not modify unselected env buffers.
- Global `write_index` does not change because of partial reset.
- `max_delay_steps` is fixed at environment construction. It cannot change at runtime.

First six executed squashed actions after reset for policy-issued sequence `a0, a1, a2, a3, a4, a5`, where `0` is the reset zero action:

| Policy step | delay=0 | delay=1 | delay=2 | delay=3 |
| ---: | --- | --- | --- | --- |
| 0 | a0 | 0 | 0 | 0 |
| 1 | a1 | a0 | 0 | 0 |
| 2 | a2 | a1 | a0 | 0 |
| 3 | a3 | a2 | a1 | a0 |
| 4 | a4 | a3 | a2 | a1 |
| 5 | a5 | a4 | a3 | a2 |

This table is the required no-off-by-one reference for unit tests.

## Previous Action Semantics

The Actor field `previous_squashed_action` means:

```text
the previous policy-issued squashed action
```

It does not mean:

```text
the delayed action actually executed by the dynamics
```

Reasons:

1. It preserves the M5, M6, and M7A Actor field meaning.
2. The previous policy-issued command is deployable information.
3. The executed action depends on hidden action delay.
4. Feeding executed action to the Actor would reveal additional execution-chain information.
5. M7B tests robustness to unknown execution delay.

Implementation must maintain two independent tensors:

```text
previous_squashed_action
executed_squashed_action
```

Usage:

- `previous_squashed_action` enters the next Actor observation.
- `executed_squashed_action` is used for dynamics, internal state, diagnostics, and audit.
- `executed_squashed_action` does not enter the Actor.
- The Critic remains the fixed 65D layout and does not add executed action.

Tests must verify that `previous_squashed_action` and `executed_squashed_action` differ when `delay > 0`, that the Actor slice uses previous policy-issued action, and that both tensors are equal in M7B-S0 when `delay=0`.

## Wind and Gust Model

The first formal M7B round uses horizontal wind only:

```text
wind_direction_mode = "horizontal"
direction = [cos(theta), sin(theta), 0]
theta ~ uniform[0, 2*pi)
```

3D spherical wind is not used in first formal M7B.

Steady wind:

- Sample once per episode per env.
- Hold fixed until the env resets.
- Reset only resamples selected envs.

Per-env gust state:

```text
gust_target_w
gust_current_w
gust_remaining_policy_steps
gust_segment_index
```

Gust target updates only at policy-step boundaries. Gust current is low-pass updated at every physics substep:

```text
alpha = 1 - exp(-physics_dt / gust_tau_s)
gust_current_w = gust_current_w + alpha * (gust_target_w - gust_current_w)
```

Default:

```text
gust_tau_s = 0.20 s
```

Config validation requires `gust_tau_s > 0`.

Reset default:

```text
gust_target_w = newly sampled gust target
gust_current_w = gust_target_w
gust_remaining_policy_steps = sampled duration
gust_segment_index = 0
```

Segment counter semantics:

- Duration `N` means the same target acts for exactly `N` policy steps.
- Decrement `gust_remaining_policy_steps` exactly once per policy step.
- Do not decrement in physics substeps.
- When remaining becomes zero, the next policy-step start samples a new segment.
- The target remains fixed within a policy step.
- The current gust may smooth at physics-substep frequency.

For duration `3`, the target sequence is:

| Policy step | remaining at step start | target action | remaining after policy-step accounting |
| ---: | ---: | --- | ---: |
| 0 | 3 | use target 0 | 2 |
| 1 | 2 | use target 0 | 1 |
| 2 | 1 | use target 0 | 0 |
| 3 | 0 | sample target 1, use target 1 | duration_1 - 1 |
| 4 | duration_1 - 1 | use target 1 | duration_1 - 2 |

Wind acceleration used by dynamics:

```text
wind_acceleration_w = steady_wind_w + gust_current_w
```

Do not separately clip the wind sum. The final `a_applied_abs_max` clamp is the safety boundary for combined tracking, drag, and wind.

## Stateless RNG

M7B dynamics, delay, steady wind, and gust must use stateless counter-based deterministic sampling. A single global `torch.Generator` is forbidden for these M7B processes.

Sampling key:

```text
base_seed
env_id
episode_count
stream_id
sample_counter
```

Requirements:

- Partial reset does not change unselected env current or future random sequences.
- `env_ids` ordering does not change samples assigned to env ids.
- One env ending early does not affect another env's future random sequence.
- Changing `num_envs` preserves common env-id sequences as much as possible.
- Dynamics, delay, steady wind, and gust use independent streams.

Required streams:

| Stream | Purpose |
| ---: | --- |
| 1001 | `tau_velocity_scale` |
| 1002 | `acceleration_limit_scale` |
| 1003 | `speed_limit_scale` |
| 1004 | `linear_drag` |
| 1005 | `action_delay_steps` |
| 1010 | `steady_wind_magnitude` |
| 1011 | `steady_wind_angle` |
| 1020 | `gust_duration` |
| 1021 | `gust_magnitude` |
| 1022 | `gust_angle` |

Episode-fixed parameters use:

```text
episode_count as episode key
sample_counter = 0
```

Gust segments use:

```text
episode_count as episode key
```

The existing target-motion library may continue using its existing generator because M7B must not modify target motion behavior. M7B randomization streams must be separate from target-motion streams.

## Actor and Critic Boundaries

Actor remains 25D:

```text
0:3    p_rel_obs_w
3:6    v_rel_obs_w
6:9    v_ego_w
9:15   R_ego_6d
15:18  omega_ego_b
18:21  previous policy-issued squashed action
21:24  b_des_w
24:25  d_offset
```

Actor must not receive:

- `executed_squashed_action`
- dynamics parameters
- action delay
- wind or gust truth
- gust counters
- target mode
- target generator parameters
- target acceleration truth
- future target states
- future target commands
- future target schedules
- future wind or gust schedules

Critic is fixed 65D:

```text
0:25    actor_obs
25:28   p_ego_w
28:31   p_target_w
31:34   v_target_w
34:37   a_target_w
37:43   R_target_6d
43:46   omega_target_b
46:50   mode_one_hot
50:56   target_motion_current_params
56:57   episode_phase
57:58   tau_velocity_scale
58:59   acceleration_limit_scale
59:60   speed_limit_scale
60:61   linear_drag
61:62   normalized_action_delay_steps
62:65   current_wind_acceleration_w
```

Delay normalization:

```text
normalized_action_delay_steps = action_delay_steps / max(1, max_delay_steps)
```

Implementation requirements:

- Create a dedicated `assemble_critic_observation_m7b` helper.
- Do not modify the existing M5-M7A 57D `assemble_critic_observation` behavior.
- Unit tests must verify every slice by value.
- GRU and feedforward M7B tasks use identical 65D Critic observations.
- Critic may receive current `wind_acceleration_w` only.
- Critic must not receive future gust targets, future schedules, gust remaining steps, or future target states.

## RSL-RL Contract Updates

M7B implementation must update training/play/evaluate contract checks without breaking M6/M7A:

- `Isaac-Uav-Rendezvous-M7B-GRU-v0` is a recurrent task.
- M6 and M7A recurrent critic input dimension remains `57`.
- M7B recurrent critic input dimension is `65`.
- All recurrent Actor input dimensions remain `25`.
- All action dimensions remain `3`.
- Checkpoint shape incompatibility must fail quickly.
- A 57D M7A checkpoint must not be silently loaded into a 65D M7B runner.

M7A zero-shot comparison must use an explicit compatibility path:

1. Run the M7A task/policy directly through a deliberate M7B disturbance evaluation wrapper, or
2. Load only compatible Actor components through an explicitly documented Actor-only baseline path.

Do not pretend a 57D Critic checkpoint is directly compatible with a 65D M7B training runner.

## Experiment Stages

| Experiment stage | Dynamics randomization | Delay | Wind/gust | Training target |
| --- | --- | --- | --- | --- |
| M7B-S0 Clean regression | scales `1`, drag `0` | `0` | `0` | Feedforward only |
| M7B-S1 Dynamics only | tau/accel/speed/drag | `0` | `0` | Feedforward |
| M7B-S2 Delay only | nominal | integer `[0,3]` | `0` | Feedforward |
| M7B-S3 Wind only | nominal | `0` | steady + gust | Feedforward |
| M7B-S4 Combined | tau/accel/speed/drag | integer `[0,3]` | steady + gust | Feedforward, optional GRU ablation |

M7B-S0 strict config:

```text
tau_velocity_scale = 1.0
acceleration_limit_scale = 1.0
speed_limit_scale = 1.0
linear_drag = 0.0
action_delay_steps = 0
steady_wind_w = 0
gust_target_w = 0
gust_current_w = 0
clean M7A observation pipeline
v_max unchanged
physical speed limit = nominal v_abs_max
reward unchanged
termination unchanged
target distribution unchanged from M7A/M6 mixed task
collision-risk rule unchanged
```

M7B-G2: nominal dynamics equivalence before M7B-S1:

1. Pure tensor helper single-step equivalence to existing M5-M7A formula with absolute error `<= 1e-6`.
2. Fixed action sequence CPU helper equivalence over 1000 steps with max error `<= 1e-6`.
3. Isaac runtime equivalence uses existing sync tolerances, suggested max error `<= 1e-5`.
4. Clean validation success `>= 0.98`.
5. Collision risk `0`.
6. Any material offset or relative-speed degradation must stop progression to M7B-S1 until explained.

## M7B-S4 Curriculum

M7B-S4 may use curriculum over randomization strength only:

| Iterations | Range scale |
| ---: | ---: |
| 0-99 | 0% to 40% |
| 100-199 | 40% to 70% |
| 200-299 | 70% to 100% |

The curriculum may scale randomized parameter ranges and wind/gust magnitudes only. It must not alter reward, success thresholds, termination conditions, collision-risk definition, Actor observations, target-motion probabilities, or `d_safe`.

All formal evaluations use 100% official ranges, not curriculum-reduced ranges.

## Training Strategy

Primary training task:

```text
Isaac-Uav-Rendezvous-M7B-Feedforward-v0
```

Initial budget:

```text
num_envs = 256
num_steps_per_env = 128
iterations = 300
seed = 42
```

M7B-S0, M7B-S1, M7B-S2, and M7B-S3 must complete infrastructure checks and training before M7B-S4 combined training begins. If 300 iterations are unstable, analyze failure mode, saturation, delay distribution, wind statistics, and range severity before any extension. Maximum extension is 600 iterations.

GRU rules:

- No M7B-S0 GRU retraining required.
- No full GRU M7B-S1 through M7B-S3 comparison required.
- Only one matched-budget M7B-S4 GRU ablation is allowed after the M7B-S4 feedforward candidate is stable.
- The GRU ablation is not trained until it beats feedforward and is not the default candidate.

## Formal Validation Sets

Every formal checkpoint evaluation uses an independent process and validation seed, such as `4242`. Each validation set uses at least 512 episodes and balanced target modes:

```text
25% ConstantVelocity
25% ConstantAcceleration
25% ConstantTurn
25% PiecewiseAcceleration
```

Required sets:

1. Clean nominal: all M7B-S0 nominal settings.
2. In-distribution randomization: random draws from formal stage ranges.
3. Edge-of-range validation: deterministic presets below.
4. Combined M7B: full M7B-S4 ranges.
5. M7A zero-shot baseline under M7B combined disturbances through an explicit compatibility path.
6. Optional M7B-S4 GRU ablation.

Metrics:

- overall success
- per-mode success
- collision risk
- return
- offset p95
- relative-speed p95
- convergence time
- episode timeout
- action saturation
- tracking acceleration saturation
- total acceleration safety saturation
- speed saturation
- workspace violation
- height violation
- finite state

## Edge-of-Range Presets

Edge validation must include deterministic presets, not only random sampling:

| Preset | Parameter values |
| --- | --- |
| nominal | all M7B-S0 values |
| minimum tau | `tau_velocity_scale=0.75` |
| maximum tau | `tau_velocity_scale=1.50` |
| minimum acceleration authority | `acceleration_limit_scale=0.80` |
| maximum acceleration authority | `acceleration_limit_scale=1.20` |
| minimum physical speed limit | `speed_limit_scale=0.90` |
| maximum physical speed limit | `speed_limit_scale=1.10` |
| maximum drag | `linear_drag=0.15` |
| maximum delay | `action_delay_steps=3` |
| maximum steady wind | `steady_wind_magnitude=0.20*a_max_nominal` |
| maximum gust | `gust_magnitude=0.15*a_max_nominal` |
| combined worst-safe edge | tau min, accel min, speed min, max drag, delay 3, max wind, max gust, with final `a_applied_abs_max` and safety limits active |

Each preset must record exact parameter values, validation seed, episode count, and all formal metrics. Presets must not exceed `a_applied_abs_max`, speed, workspace, height, or non-contact safety limits.

## Robustness Comparison

At minimum compare:

1. M7A accepted feedforward checkpoint zero-shot under M7B combined disturbances.
2. M7B robust feedforward checkpoint.
3. Optional M7B-S4 GRU checkpoint.

Robust training value is proven only if M7B robust feedforward improves over M7A zero-shot on Combined M7B by at least one of:

- success improves by at least 10 percentage points
- offset p95 decreases by at least 15%
- relative-speed p95 decreases by at least 15%
- timeout materially decreases

while:

- collision risk does not increase
- clean success decreases by no more than 2 percentage points

If these acceptance criteria are not met, report a limitation. Do not hide negative results and do not weaken safety or success criteria.

## Required Tests

Expected values must be independently derived and must not call the production function under test.

Required test files:

```text
tests/test_m7b_action_delay.py
tests/test_m7b_dynamics_equation.py
tests/test_m7b_parameter_sampling.py
tests/test_m7b_wind_process.py
tests/test_m7b_partial_reset.py
tests/test_m7b_rng_reproducibility.py
tests/test_m7b_actor_isolation.py
tests/test_m7b_critic_layout.py
tests/test_m7b_task_registration.py
tests/test_m7b_fair_ablation.py
tests/test_m7b_noncontact_objective.py
```

Coverage requirements:

1. FIFO impulse response for delays 0, 1, 2, and 3.
2. FIFO reset clears selected env slots.
3. `previous_squashed_action` differs from `executed_squashed_action` when `delay > 0`, and Actor uses previous issued action.
4. Dynamics one-step equation with independent expected value.
5. Total acceleration cap.
6. Speed cap.
7. M7B-S0 1000-step helper equivalence to M5-M7A nominal dynamics.
8. Gust duration off-by-one behavior.
9. Gust physics-substep low-pass filter formula.
10. Gust reset initialization.
11. RNG reset-order invariance.
12. RNG partial-reset future invariance.
13. Critic 65D slice values.
14. Actor isolation from dynamics, delay, wind, gust, mode, and future data.
15. Non-contact invariant and collision-risk termination/reward behavior unchanged.
16. M7B recurrent 65D contract.
17. Deterministic edge presets.
18. Existing M2-M7A task registrations and entry points unchanged.

## Runtime Audits

Add `scripts/audit_m7b_dynamics.py` with at least:

- Actor observation shape 25.
- Critic observation shape 65.
- Action shape 3.
- Policy class.
- Dynamics parameter sampled ranges.
- Delay distribution.
- FIFO causality.
- `previous_squashed_action` and `executed_squashed_action` semantics.
- Steady wind statistics.
- Gust target/current statistics.
- Tracking acceleration saturation.
- Total acceleration safety saturation.
- Speed saturation.
- Workspace violation.
- Height violation.
- Collision risk.
- RigidObject synchronization error.
- Partial reset no crosstalk.
- 10000-step finite rollout.
- No NaN, Inf, or CUDA error.

Add `scripts/evaluate_m7b_robustness.py` for formal reports. It must not create multiple full Isaac environments in one process. Each checkpoint or stage evaluation must run in an independent process.

Before formal M7B training, explicitly deprecate `scripts/audit_m7_pomdp_comparison.py`:

- It must fail at startup with a clear error.
- The error must say same-process Isaac lifecycle results are not trusted for formal metrics.
- The error must point to `scripts/evaluate.py` or the new independent-process M7B evaluation entry point.

## Regression Verification

Before user acceptance, rerun and document:

1. Syntax compile.
2. Full pytest.
3. M2 audit.
4. M3 audit.
5. M4 formal audit.
6. M5 runtime audit.
7. M6 recurrent runtime audit.
8. M7A observation pipeline audit.
9. M7A accepted feedforward clean validation.
10. M7B dynamics audit 10000 steps.
11. M7B-S0 validation.
12. M7B-S1 validation.
13. M7B-S2 validation.
14. M7B-S3 validation.
15. M7B-S4 validation.
16. M7A zero-shot vs M7B robust comparison.
17. Optional GRU M7B-S4 ablation.
18. `git diff --check`.

## Planned Files

The plan permits later implementation edits only after final plan approval. Expected implementation files are:

New source/config files:

```text
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7b_env.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7b_env_cfg.py
source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/__init__.py
source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/action_delay.py
source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/m7b.py
source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/stateless_rng.py
source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/wind.py
```

Modified source/scripts after approval:

```text
source/uav_rendezvous_rl/uav_rendezvous_rl/mdp/rendezvous.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/__init__.py
source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py
scripts/train.py
scripts/evaluate.py
scripts/audit_m7_pomdp_comparison.py
```

New tests/scripts/docs after approval:

```text
tests/test_m7b_action_delay.py
tests/test_m7b_dynamics_equation.py
tests/test_m7b_parameter_sampling.py
tests/test_m7b_wind_process.py
tests/test_m7b_partial_reset.py
tests/test_m7b_rng_reproducibility.py
tests/test_m7b_actor_isolation.py
tests/test_m7b_critic_layout.py
tests/test_m7b_task_registration.py
tests/test_m7b_fair_ablation.py
tests/test_m7b_noncontact_objective.py
scripts/audit_m7b_dynamics.py
scripts/evaluate_m7b_robustness.py
docs/m7b_verification.md
```

This revision does not authorize implementation, commits, pushes, `review/m7b`, `m7b-accepted`, or M7C.
