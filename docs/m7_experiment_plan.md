# M7A Experiment Plan

Date: 2026-07-23

Status: planned, implementation not yet started.

## Scope

M7A tests whether policy history has measurable task value when the Actor receives strictly causal, degraded deployable observations. M7A is limited to controlled partial observability: observation delay, relative-velocity low-frequency updates with sample-and-hold, observation dropout with last-valid-value hold, and small zero-mean Gaussian observation noise.

M7A does not authorize M7B or M7C. It does not add dynamics randomization, wind, control execution delay, mass/inertia/thrust randomization, Crazyflie, Multirotor/Thruster, Pegasus, PX4, ROS 2, cameras, image input, visual networks, or distance-dependent visual error models.

## Motivation

M6 recurrent training and hidden-state management are functional, but a measurable implicit-prediction advantage over the fair feedforward baseline was not demonstrated.

The M6 Actor observed current `p_rel_w` and `v_rel_w`, making the state close to Markov for the accepted simplified dynamics and target-motion ranges. Under that setting, a feedforward policy could already infer most control-relevant state from the current observation, and the GRU did not convert history sensitivity into better validation performance.

M7A deliberately introduces controlled partial observability while keeping the task, dynamics, rewards, terminations, and deployable observation dimension fixed. If history is valuable, the GRU should outperform a fair feedforward policy under the same causal observation degradation.

## Tasks

M7A adds only new independent tasks:

```text
Isaac-Uav-Rendezvous-M7A-GRU-v0
Isaac-Uav-Rendezvous-M7A-Feedforward-v0
```

The accepted tasks below must not change behavior:

```text
Isaac-Uav-Rendezvous-Direct-v0
Isaac-Uav-Rendezvous-Baseline-v0
Isaac-Uav-Rendezvous-RL-v0
Isaac-Uav-Rendezvous-Recurrent-v0
Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0
```

## Information Boundary

The M7A Actor remains 25D:

| Field | Dim | Source |
| --- | ---: | --- |
| `p_rel_obs_w` | 3 | causal observation pipeline |
| `v_rel_obs_w` | 3 | causal observation pipeline |
| `v_ego_w` | 3 | current deployable ego state |
| `R_ego_6d` | 6 | current deployable ego state |
| `omega_ego_b` | 3 | current deployable ego state |
| `previous_squashed_action` | 3 | previous action after `tanh` |
| `b_des_w` | 3 | desired offset |
| `d_offset` | 1 | offset magnitude |

Only the relative position and relative velocity entries are replaced by observation-pipeline outputs. The Actor must not receive `p_rel_truth_w`, `v_rel_truth_w`, `a_target_w`, mode labels, target generator parameters, dropout masks, observation age, raw history buffers, future target states, future commands, future segment schedules, or complete future trajectories.

The Critic remains 57D and may use current truth and current observation-degradation state if needed, but it must not use future states. To preserve the accepted 57D critic contract and comparison with M6, the first M7A implementation keeps the critic dimension at 57D.

## Observation Pipeline

The planned pure-PyTorch observation package is:

```text
source/uav_rendezvous_rl/uav_rendezvous_rl/observations/
    __init__.py
    configs.py
    history_buffer.py
    corruption.py
    pipeline.py
```

Design requirements:

- Batch all environment operations.
- Avoid per-environment Python loops on the step path.
- Support partial reset by `env_ids`.
- Separate immutable configuration from runtime state.
- Use an independent, reproducible `torch.Generator`.
- Keep degradation strictly causal.

Each policy step uses this order:

1. Read current simulation truth.
2. Generate current measurement sample.
3. Write current sample into history buffers.
4. Read current or past samples according to configured delay.
5. Apply low-frequency update and sample-and-hold.
6. Apply dropout and last-valid-value hold.
7. Apply zero-mean Gaussian noise.
8. Build Actor observation.
9. Policy produces action.
10. Environment advances.

Delay semantics are fixed:

| Delay | Output |
| ---: | --- |
| `0` | current measurement sample |
| `1` | previous observation-cycle sample |
| `N` | sample from `N` observation cycles ago |

At startup and reset, the full buffer is filled with the current initial measurement so no uninitialized or previous-episode sample can leak into the Actor.

## Experiment Stages

| Stage | Position observation | Velocity observation | Dropout | Noise | Formal training in M7A |
| --- | --- | --- | --- | --- | --- |
| 0 Clean regression | delay `0`, 50 Hz | delay `0`, 50 Hz | `0` | `0` | yes |
| 1 Velocity low-frequency | delay `0`, 50 Hz | delay `0`, 10 Hz sample-and-hold | `0` | `0` | yes |
| 2 Medium delay | delay `1`, 50 Hz | delay `3`, 50 Hz | `0` | `0` | yes |
| 3 Dropout infrastructure | delay `0`, 50 Hz | delay `0`, 50 Hz | position `5%`, velocity `10%` | `0` | no, audit only |
| 4 Combined infrastructure | delay `1`, 50 Hz | delay `3`, 10 Hz | position `5%`, velocity `10%` | small Gaussian | no, audit only |

Stage 0 must show the observation pipeline itself does not introduce obvious degradation relative to the M6 trend. Stages 1 and 2 are the formal partial-observability tests for history value. Stages 3 and 4 are implemented and audited for stability and causality but are not full-training requirements in this round.

## Fair GRU vs Feedforward Comparison

For each formal stage, train and validate both tasks with identical settings except policy class:

| Item | Required match |
| --- | --- |
| Environment dynamics | identical |
| Target mode distribution | identical |
| Reward and termination | identical |
| Actor and Critic dimensions | identical 25D / 57D |
| Observation degradation config | identical |
| Seed and interaction budget | identical |
| Rollout length | identical |
| Normalization | identical |
| Validation split and seed | identical |
| Evaluation episode count | identical |

Initial formal budget is `num_envs=256`, `num_steps_per_env=128`, `iterations=300`, `seed=42`. If a 300-iteration pair is unstable, analyze before deciding whether to extend, with an upper limit of 600 iterations.

## Acceptance Criteria

M7A implementation acceptance requires:

- New M7A tasks registered without changing M2 through M6 task behavior.
- Actor observation shape remains `25` and contains only deployable, causal fields.
- Critic observation shape remains `57` and has no future leakage.
- Delay, sample-and-hold, dropout, noise, and partial reset tests pass.
- Pulse and ramp tests prove no future leakage and no off-by-one delay error.
- Stage 3 and Stage 4 observation configurations pass implementation tests and a 10000-step finite runtime audit.
- M2 through M6 regression audits pass.
- Formal Stage 0, Stage 1, and Stage 2 GRU/Feedforward pairs complete training and validation under matched budgets.
- Safety must not regress: collision-risk rate must remain zero for accepted formal comparisons.

History value may be claimed only if at least one of these holds without safety degradation:

- GRU success improves by at least `5` percentage points.
- GRU successful offset p95 decreases by at least `10%`.
- GRU successful relative-speed p95 decreases by at least `10%`.
- GRU average return is significantly higher.
- GRU convergence time decreases by at least `10%`.

History sensitivity alone is not performance evidence.

## Verification Outputs

M7A will produce:

- Unit tests for delay, sample-and-hold, dropout, noise seed reproducibility, partial reset, no future leakage, actor isolation, task registration, and fair ablation.
- `scripts/audit_m7_observation_pipeline.py` for pipeline metrics, partial reset, no future leakage, finite rollout, and M2-M6 regression entry points.
- `scripts/audit_m7_pomdp_comparison.py` for same-config GRU vs feedforward comparison summaries.
- `docs/m7a_verification.md` recording design, tests, audits, formal Stage 0/1/2 results, whether history performance advantage is proven, warnings, and failed items.

M7B and M7C remain not authorized after M7A unless the user explicitly approves them.
