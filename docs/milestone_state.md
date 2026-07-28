# Milestone State

Current milestone: M7
Current sub-milestone: M7B
Status: in_progress
Next executable stage: M7B-S1
M7C authorization: not authorized
Authoritative M7B execution order: M7B-S0 clean, M7B-S1 dynamics only, M7B-S2 control delay only, M7B-S3 wind only, M7B-S4 combined
M7B-S1 status: completed; stop before M7B-S2 pending audit/user confirmation
Last completed milestone: M7A
M4 accepted tag: m4-accepted
M4 accepted commit: 36592b6a14cd1a00d6bb689b3a33d27fe610a3b1
M5 implementation commit: 887bb20a3d5a44eac479fc451fab89aa18296b57
M5 accepted tag: m5-accepted
M5 accepted commit: 61e3a8107b966bf146b46a3855b0ac256cdf53c2
M5 independent audit result: ACCEPT M5 WITH NON-BLOCKING ISSUES
M6 implementation commit: 2f4dd9c85b931075294f59bafe7e39d9b2127765
M6 accepted tag: m6-accepted
M6 accepted commit: acc27beca2528db21fe1604118e448a87f7e298a
M6 acceptance result: ACCEPT M6 WITH MAJOR LIMITATION
M6 independent audit result: ACCEPT M6 WITH MAJOR LIMITATION
M7A implementation commit: 348d0ba1782eedc61c692b5e0558dec04104abab
M7A accepted tag: m7a-accepted
M7A acceptance result: ACCEPT M7A WITH MAJOR LIMITATION
M7A independent audit result: ACCEPT M7A WITH MAJOR LIMITATION
M7B-G5 independent audit result: ACCEPT M7B-G5 WITH NON-BLOCKING ISSUES
M7B-S0 checkpoint SHA-256: 7254b343972ab30a27c48dcb7fd5c5a90079ef6be25c067ec92cc0679f32e5f8
M7B-S1 checkpoint SHA-256: 2e159913068a795eef6b48924267b453876f547d6113cffc3c69efaac61de3dc
Primary M7B policy baseline: feedforward
Secondary ablation: GRU

## M6 Acceptance Summary

M6 accepted capabilities:

- Recurrent PPO infrastructure
- GRU Actor and Critic memory
- Per-env done-mask hidden reset
- Recurrent play and evaluation
- Checkpoint save/load/resume
- History sensitivity
- Mixed-mode safe operation
- Fair feedforward ablation

Major limitation:

A measurable performance advantage of GRU over the fair feedforward baseline was not demonstrated.

Independent audit:

```text
docs/m6_independent_audit.md
```

Historical pre-M7B note: at M7A acceptance time, M7B and M7C were not authorized. Current M7B status is governed by the M7B Authorization section below; M7C remains not authorized.

## M7A Authorization

Authorized M7A work:

- Controlled observation delay.
- Relative-velocity low-frequency update and sample-and-hold.
- Observation dropout with last-valid-value hold.
- Small zero-mean Gaussian observation noise.
- Independent GRU and feedforward fair-control tasks.
- Observation history buffer.
- Causality and no-future-leakage tests.
- M7A training and validation for stages 0, 1, and 2.
- Infrastructure, tests, and 10000-step stability audits for stages 3 and 4.
- M2 through M6 regression tests.

Forbidden in M7A:

- M7B dynamics randomization.
- Wind disturbance.
- Control execution delay.
- Mass, inertia, or thrust randomization.
- Crazyflie, Multirotor/Thruster, Pegasus, PX4, ROS 2.
- First-stage vision network, image input, or distance-dependent visual error model.
- Modifying accepted M2 through M6 task behavior.
- Adding mode labels, target motion parameters, target acceleration truth, target future state, target future command, future segment schedule, complete future trajectory, dropout mask, observation age, or raw history buffer contents to the Actor.
- Entering M7B or M7C.

## M6 Authorized Work

- 独立Recurrent RL任务 `Isaac-Uav-Rendezvous-Recurrent-v0`
- GRU Recurrent PPO
- 非对称循环Actor-Critic
- 25维可部署Actor观测，不含显式目标运动模式、生成器参数或未来状态
- 57维特权Critic观测
- hidden state与done mask管理
- checkpoint save/resume验证
- recurrent play/evaluate
- 受控混合目标运动训练：ConstantVelocity、ConstantAcceleration、ConstantTurn、PiecewiseAcceleration
- 公平前馈ablation对照
- 隐式目标运动历史利用证据

## Forbidden In M6

- 显式目标未来轨迹
- B样条轨迹输出
- 风扰动
- 动力学随机化
- 观测噪声
- 观测延迟
- 丢帧
- 距离相关感知误差
- Crazyflie
- Multirotor/Thruster
- Pegasus
- PX4
- ROS 2
- 相机或第一阶段感知网络
- 目标未来状态
- 目标未来指令
- 未来segment schedule进入Actor
- motion mode或生成器参数进入Actor

## Notes

- M4 accepted by user; acceptance tag is `m4-accepted` at commit `36592b6a14cd1a00d6bb689b3a33d27fe610a3b1`.
- M5 starts from `m4-accepted` on branch `feature/m5`.
- M5 accepted by user with tag `m5-accepted` at commit `61e3a8107b966bf146b46a3855b0ac256cdf53c2`.
- M6 starts from `m5-accepted` on branch `feature/m6` after explicit user authorization.
- The original `Isaac-Uav-Rendezvous-Direct-v0` task must remain an M2/M3 regression task with stationary ego and no-op action.
- The original `Isaac-Uav-Rendezvous-Baseline-v0` task must remain the M4 deterministic baseline and must not be affected by RL actions.
- The existing `Isaac-Uav-Rendezvous-RL-v0` task must remain the M5 feedforward PPO task and must not be converted into a recurrent task.
- M7A starts from `m6-accepted` on local branch `feature/m7` after explicit user authorization.
- Historical pre-M7B note: M7B was not authorized when this note was written; M7B is now authorized and in progress. M7C is not authorized.

## M5 Implementation Snapshot

- Added independent task `Isaac-Uav-Rendezvous-RL-v0`.
- Implemented `Box(-inf, inf, shape=(3,), dtype=float32)` raw action with environment-owned `tanh` mapping to `v_cmd_w`.
- Implemented simplified acceleration-limited ego dynamics with default `v_max=3.0`, `v_abs_max=5.0`, `a_max=2.0`, and `tau_v=0.25`.
- Implemented per-episode sampled horizontal `b_des_w` with fixed `d_offset=5.0`.
- Implemented 25D Actor observation and 57D privileged Critic observation under `{"policy": ..., "critic": ...}`.
- Implemented separated reward terms and explicit termination accounting.
- Added feedforward asymmetric RSL-RL PPO config with `obs_groups={"policy": ["policy"], "critic": ["critic"]}` and `clip_actions=None`.
- Added M5 train, deterministic play, deterministic evaluate, and runtime audit scripts.
- Kept M6 recurrent PPO, GRU, and LSTM out of scope.

## M5 Verification

- Final verification details are recorded in `docs/m5_verification.md`.
- `py_compile` for changed M5 env/evaluate files passed.
- `pytest tests -q` passed: `55 passed`.
- M5 64-env, 10000-step audit passed with `--scenario all --seed 42 --split train --device cuda:0 --headless` after final reward-accounting fix.
- M5 oracle audit path reported `success_rate=1.0`, `collision_risk_count=0`, `height_violation_count=0`, `workspace_violation_count=0`, and `speed_limit_count=0`.
- M5 zero/random audit paths remained finite and reset on unsafe safety/height terminations as expected for non-controller/random-controller policies.
- M5 PPO startup passed for `--num_envs 64 --max_iterations 5 --seed 42 --run_name m5_startup_recheck` and produced checkpoint `logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_18-44-26_m5_startup_recheck/model_4.pt`.
- Final M5 feedforward PPO training passed for `--num_envs 256 --max_iterations 300 --seed 42 --run_name m5_rewardfix_300_seed42` and produced checkpoint `logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_19-04-26_m5_rewardfix_300_seed42/model_299.pt`.
- Deterministic validation of `model_299.pt` on validation split with seed `4242`, 64 envs, and 4 episodes passed: `success_rate=1.0`, `collision_risk_rate=0.0`, `workspace_violation_rate=0.0`, `height_violation_rate=0.0`, `speed_violation_rate=0.0`, `success_offset_error.p95=0.3593`, and `success_relative_speed.p95=0.1695`.
- M2, M3, and M4 regression audits were rerun during M5 work and passed.

## M5 Final Validation

- Split: validation
- Seed: 4242
- Environments: 64
- Episodes per environment: 4
- Total episodes: 256
- Success rate: 1.0
- Collision risk rate: 0.0
- Successful offset error p95: 0.3593 m
- Successful relative speed p95: 0.1695 m/s

Local checkpoint, not tracked by Git:

```text
logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_19-04-26_m5_rewardfix_300_seed42/model_299.pt
```

Non-blocking issues are recorded in:

- `docs/known_issues.md`
- `docs/m5_independent_audit.md`

## M6 Implementation Snapshot

- Added independent recurrent task `Isaac-Uav-Rendezvous-Recurrent-v0` using `UavRendezvousRecurrentEnv`, which subclasses the accepted M5 RL environment without converting `Isaac-Uav-Rendezvous-RL-v0` into a recurrent task.
- Added M6 mixed target-motion config with `ConstantVelocity`, `ConstantAcceleration`, `ConstantTurn`, and `PiecewiseAcceleration` at `25%` each.
- Kept the Actor observation contract at 25D and Critic observation contract at 57D; the M6 env subclass does not add Actor inputs or expose motion mode/generator parameters to the Actor.
- Added GRU PPO config `UavRendezvousRecurrentPPORunnerCfg` with `ActorCriticRecurrent`, `rnn_type="gru"`, `rnn_hidden_dim=128`, `rnn_num_layers=1`, `num_steps_per_env=128`, independent actor/critic observation normalization, `clip_actions=None`, and asymmetric obs groups.
- Added fair feedforward ablation config and task `Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0` over the same M6 mixed-mode environment.
- Updated `train.py` to assert and print M6 recurrent startup contract: policy class, `is_recurrent`, actor/critic GRU memories, 25D policy input, 57D critic input, and 3D action dim.
- Updated `play.py` and `evaluate.py` to use RSL-RL `act_inference()` and to call `policy.reset(dones)` after every step; reset calls are made under `torch.inference_mode()` to match RSL-RL rollout behavior for inference tensors.
- Added recurrent hidden-state audit support to `play.py` via `--audit_hidden_state`.
- Added per-mode evaluation summary and deterministic inference repeat check to `evaluate.py` via `summary_by_mode` and `--determinism_check`.
- Added runtime audits `scripts/audit_m6_recurrent_runtime.py` and `scripts/audit_m6_history_sensitivity.py`.
- Added M6 tests for config, registration, hidden reset, partial done masks, checkpoint state dict round-trip, recurrent evaluation loop checks, and Actor isolation.

## M6 Verification

Final verification details are recorded in `docs/m6_verification.md`.

- `git diff --check` passed.
- `py_compile` for changed M6 source and scripts passed.
- Full test suite passed: `68 passed`.
- M2 runtime regression passed with `audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42`.
- M3 target-motion runtime regression passed with `audit_m3_motion_runtime.py --num_envs 16 --steps 5000 --seed 42 --split train`.
- M4 deterministic baseline regression passed with `audit_m4_baseline_runtime.py --num_envs 64 --episodes 5 --seed 42 --split train`.
- M5 runtime regression passed with `audit_m5_rl_runtime.py --scenario all --num_envs 64 --steps 10000 --seed 42 --split train`; the oracle path achieved `success_rate=1.0` with zero collision-risk, workspace, height, and speed-limit counts.
- M5 trained validation regression reproduced accepted metrics on validation split, seed `4242`, 64 envs, 4 episodes/env: `success_rate=1.0`, `collision_risk_rate=0.0`, `success_offset_error.p95=0.3592815101`, and `success_relative_speed.p95=0.1695371419`.
- M6 recurrent runtime audit passed with 8 envs and 64 steps. It verified `ActorCriticRecurrent`, `is_recurrent=True`, actor GRU, critic GRU, actor input `25`, critic input `57`, action dim `3`, independent actor/critic memories, partial done hidden reset with `done_count=1` and `kept_count=7`, checkpoint-schema load, and finite rollout.
- M6 recurrent deterministic play passed with `--audit_hidden_state`, 4 envs, 2 steps, seed `4242`, and the final mixed GRU checkpoint.
- CV-only GRU formal training passed the CV gate on validation split, seed `4242`, 64 envs, 4 episodes/env: `success_rate=0.9921875`, `collision_risk_rate=0.0`, successful offset p95 about `0.452 m`, and successful relative-speed p95 about `0.212 m/s`.
- Mixed-mode GRU formal validation passed with balanced validation split, seed `4242`, 64 envs, 8 episodes/env, and 128 episodes/mode: `success_rate=0.998046875`, `collision_risk_rate=0.0`, successful offset p95 `0.4626103044 m`, and successful relative-speed p95 `0.2197498679 m/s`.
- Fair feedforward ablation formal validation passed under the same mixed-mode protocol: `success_rate=1.0`, `collision_risk_rate=0.0`, successful offset p95 `0.1983923167 m`, and successful relative-speed p95 `0.1089752316 m/s`.
- Final trained history-sensitivity audit passed with 16 synthetic envs and 16 history steps: identical final Actor observations produced different GRU hidden states/actions and zero feedforward action difference.
- Formal checkpoint resume audit passed from the final mixed GRU checkpoint with 64 envs and 3 resumed PPO iterations: optimizer state restored, actor/critic hidden initial norms were `0.0`, finite losses were produced, and policy parameters changed with norm `1.1829556227`.

Local M6 final checkpoints, not tracked by Git:

```text
logs/rsl_rl/uav_rendezvous_m6_gru/2026-07-22_23-46-03_m6_cv_gru_300_seed42/model_299.pt
logs/rsl_rl/uav_rendezvous_m6_gru/2026-07-22_23-56-05_m6_mixed_gru_300_seed42/model_299.pt
logs/rsl_rl/uav_rendezvous_m6_feedforward_ablation/2026-07-23_00-11-43_m6_ff_ablation_300_seed42/model_299.pt
```

M6 recurrent training and hidden-state management are functional, but a measurable implicit-prediction advantage over the fair feedforward baseline was not demonstrated.

M6 is accepted with major limitation. M7A has since passed user acceptance with major limitation on `feature/m7`. Historical note: M7B was not authorized at that time; M7B is now authorized and in progress. M7C is not authorized.

## Next Milestone Guard

- M5 has passed user acceptance.
- M6 has passed user acceptance with major limitation.
- M7A has passed user acceptance with major limitation.
- M7B is authorized and in progress.
- Do not enter M7C without explicit user confirmation.

## M7B Authorization

Authorized M7B work:

- Simplified ego dynamics randomization (tau_velocity_scale, acceleration_limit_scale, speed_limit_scale, linear_drag).
- Control execution delay (per-env action FIFO, integer steps [0,3]).
- Steady wind and piecewise gust equivalent acceleration.
- Independent M7B feedforward and GRU tasks: `Isaac-Uav-Rendezvous-M7B-Feedforward-v0`, `Isaac-Uav-Rendezvous-M7B-GRU-v0`.
- Feedforward is the primary M7B policy; GRU is secondary ablation only.
- Clean M7A observation pipeline (delay=0, dropout=0, noise=0).
- Authoritative stage order: M7B-S0 clean, M7B-S1 dynamics only, M7B-S2 control delay only, M7B-S3 wind only, M7B-S4 combined.
- Actor: 25D deployable only; no dynamics parameters, delay, wind, or future information.
- Critic: 65D with current-time privileged dynamics (tau_velocity_scale, acceleration_limit_scale, speed_limit_scale, linear_drag, normalized action_delay_steps, current wind_acceleration_w).
- No future wind, future gust, or future target information in Critic.
- Non-contact objective: collision risk not rewarded, d_safe not weakened.
- M2 through M7A task behavior unchanged.

Forbidden in M7B:

- M7C visual estimation, image input, distance-dependent error, first-stage detection/recognition networks.
- Crazyflie, Multirotor/Thruster, Pegasus, PX4, ROS 2, real-world deployment.
- Combining observation degradation (M7A) with dynamics randomization.
- Modifying accepted M2 through M7A task behavior.
- Leaking dynamics parameters, wind truth, delay, mode_id, or future information to the Actor.
- Entering M7C.

## M7A Acceptance Summary

- Implemented independent M7A GRU and feedforward tasks without modifying accepted M2 through M6 task IDs.
- Implemented strictly causal observation degradation for Stage 0 through Stage 4 infrastructure.
- Full pytest passed before formal training: `95 passed in 1.65s`.
- M2 through M6 regression audits passed during M7A implementation.
- Stage 3 and Stage 4 M7A 10000-step observation pipeline audits passed.
- Formal Stage 0, Stage 1, and Stage 2 GRU/feedforward training runs completed with matched 300-iteration budgets.
- Matched validation results are recorded in `docs/m7a_verification.md`.
- Independent audit result: `ACCEPT M7A WITH MAJOR LIMITATION`.
- The archived independent audit is a reconstructed independent-auditor copy: `docs/m7a_independent_audit.md`.
- Snapshot recertification evidence is archived in `docs/m7a_snapshot_recertification.md`.

M7A final conclusion:

```text
M7A observation-degradation infrastructure is functional, but no history-value performance advantage was demonstrated. Feedforward outperformed GRU on Stage 0, Stage 1, and Stage 2 matched validation while preserving zero collision risk.
```

Residual M7A issue:

```text
scripts/audit_m7_pomdp_comparison.py exposed an Isaac same-process multi-environment lifecycle issue. Formal metrics were collected with scripts/evaluate.py instead.
```

## M7B Progress

### M7B-I1: pure PyTorch infrastructure

- Created `uav_rendezvous_rl/dynamics/` package.
- Implemented `stateless_rng.py`: counter-based deterministic sampling keyed by (env_id, episode_count, seed, stream, counter).
- Implemented `action_delay.py`: per-environment push-then-read FIFO storing squashed actions.
- Implemented `m7b.py`: formal dynamics integration with tracking clamp, total safety cap (4.0 m/s²), and analytic position formula.
- Implemented `wind.py`: steady wind + piecewise gust with physics-substep low-pass filter.

### M7B-I2: environment integration

- Created `UavRendezvousM7BEnvCfg` with dynamics randomization fields.
- Created `UavRendezvousM7BEnv` subclass with overridden `_pre_physics_step`, `_apply_action`, `_get_observations`, and `_reset_idx`.
- Registered `Isaac-Uav-Rendezvous-M7B-Feedforward-v0` and `Isaac-Uav-Rendezvous-M7B-GRU-v0`.
- Added `UavRendezvousM7BFeedforwardPPORunnerCfg` and `UavRendezvousM7BGRUPPORunnerCfg`.
- Added `assemble_critic_observation_m7b` (65D) to `mdp/rendezvous.py` without modifying existing 57D function.

### M7B-I3: training/play/evaluation contracts

- Updated `train.py`: added `--m7b_stage`, `_configure_m7b_stage()`, and M7B-GRU 65D recurrent contract check.
- Updated `evaluate.py`: added `--m7b_stage` and `_configure_m7b_stage()`.
- Deprecated `scripts/audit_m7_pomdp_comparison.py`: exits with error message pointing to `evaluate.py`.

### M7B-I4: tests

- Added `tests/test_m7b_action_delay.py`: 5 pure unit tests.
- Added `tests/test_m7b_dynamics_equation.py`: 5 pure unit tests.
- Added `tests/test_m7b_task_registration.py`: 3 registration preservation tests.
- Added `tests/test_m7b_critic_layout.py`: 2 layout tests.
- Updated `tests/test_env_smoke.py` for M7B milestone and file checks.

### M7B-G1: syntax and pure unit tests

- `git diff --check` passed.
- Isaac compileall passed for changed source, script, and test files.
- Full pure pytest passed: `134 passed in 1.85s`.

### M7B-G2: nominal dynamics equivalence

- Unit tests passed for nominal dynamics equivalence, parameter sampling, and wind process: `12 passed`.
- `test_m7b_one_step_nominal_is_same_as_m5_m7a` passes within `1e-6`.
- Position formula verified as `p + v*dt + 0.5*a*dt^2`.

### M7B-G3: environment and observation contracts

- M7B feedforward S0 runtime audit passed.
- M7B GRU S0 runtime audit passed on retry after an Isaac startup segfault occurred before project environment initialization.
- Runtime contracts verified Actor observation `25D`, Critic observation `65D`, action dimension `3`, clean S0 observation pipeline, and recurrent GRU `65D` critic contract for the secondary ablation task.
- Isaac startup-stage segfault note: the first M7B GRU S0 runtime-audit attempt exited during Isaac startup before project environment initialization; the retry passed. This was treated as infrastructure-level and did not modify Isaac Lab, Isaac Sim, system packages, or the driver.

### M7B-G4: runtime and regression audits

- M7B-S4 combined 10000-step runtime audit passed.
- Compact M2, M3, M4, M5, and M6 regression audits passed.
- Compact M7A GRU observation-pipeline regression passed on retry after an Isaac startup segfault occurred before project environment initialization.
- Exact M7A compact GRU retry command that passed:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/audit_m7_observation_pipeline.py --task Isaac-Uav-Rendezvous-M7A-GRU-v0 --m7a_stage 4 --num_envs 4 --steps 8 --seed 43 --device cuda:0 --headless
```

- M7A compact GRU retry result: passed with `policy_obs_dim=25`, `critic_obs_dim=57`, finite check `true`, no-future-leakage check present, hidden reset check present, partial reset check present, and `done_count=0` over 8 audit steps.
- Isaac startup-stage segfault note: the previous M7A compact GRU attempt with seed `42` exited during Isaac startup before the project audit report was produced; the seed `43` retry above passed.

### M7B-G5: clean startup and Stage 0 validation

- M7B-S0 feedforward PPO training completed with the approved startup/formal clean budget: `256` envs, `300` iterations, seed `42`.
- Final local checkpoint, not tracked by Git:

```text
logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_01-30-42_m7b_s0_ff_300_seed42/model_299.pt
```

- M7B-S0 checkpoint SHA-256: `7254b343972ab30a27c48dcb7fd5c5a90079ef6be25c067ec92cc0679f32e5f8`.

- Balanced clean validation passed on validation split, seed `4242`, `64` envs, `8` episodes/env, `512` total episodes, exactly `128` episodes per target mode.
- Clean validation metrics: `success_rate=1.0`, `collision_risk_rate=0.0`, `workspace_violation_rate=0.0`, `height_violation_rate=0.0`, `speed_violation_rate=0.0`, successful offset p95 `0.3143340051 m`, successful relative-speed p95 `0.1617015600 m/s`, deterministic inference `max_abs_delta=0.0`.
- M7B-S0 diagnostics verified nominal dynamics and clean observation pipeline: tau/accel/speed scales `1.0`, drag `0.0`, delay `0`, wind/gust `0`, Actor observation `25D`, Critic observation `65D`, finite state `true`.

Full verification details are recorded in `docs/m7b_verification.md`.

Independent G5 audit result:

```text
docs/m7b_g5_independent_audit.md: ACCEPT M7B-G5 WITH NON-BLOCKING ISSUES
```

### M7B-S1 locked execution protocol

- Current executable stage: M7B-S1 dynamics only.
- Task: `Isaac-Uav-Rendezvous-M7B-Feedforward-v0`.
- Policy: feedforward PPO only; no GRU training in this run.
- Training source: from scratch; do not resume the M7B-S0 checkpoint.
- Stage flag: `--m7b_stage 1`.
- Observation degradation: clean M7A pipeline only, with delay `0`, dropout `0`, noise `0`, and policy-frequency position/velocity updates.
- Enabled disturbance: dynamics randomization only (`tau_velocity_scale`, `acceleration_limit_scale`, `speed_limit_scale`, `linear_drag`).
- Disabled disturbances: action delay must remain `0`; steady wind and gust must remain `0`.
- Observation dimensions: Actor `25D`, Critic `65D`.
- Training budget: `256` envs, `300` iterations, seed `42`.
- Validation protocol: validation split, seed `4242`, `64` envs, `8` episodes/env, `512` total episodes, exactly `128` episodes per target mode, deterministic inference check enabled.
- Pre-registered S1 gates: collision risk rate `0`; workspace violation rate `0`; height violation rate `0`; speed violation rate `0`; finite diagnostics `true`; deterministic `max_abs_delta=0`; overall success rate at least `0.95`; each target-mode success rate at least `0.90`; successful offset p95 at most `0.50 m`; successful relative-speed p95 at most `0.30 m/s`.

No commits, no pushes, no M7C entry. M7B-S2 through M7B-S4 formal robustness training and validation are not started.

### M7B-S1: dynamics-only training and validation

- Preflight passed: `git diff --check`, Isaac compileall, and targeted M7B tests (`41 passed in 0.92s`).
- Short S1 startup audit passed with Actor `25D`, Critic `65D`, finite diagnostics, dynamics randomization active, action delay `0`, and wind/gust `0`.
- M7B-S1 10000-step runtime audit passed on exact-command retry after an Isaac startup-stage segfault with `appState='startup'` and `UptimeSeconds='0'`; the failed attempt occurred before project environment initialization.
- Validation-split stage-isolation check confirmed S1 dynamics randomization on validation split with action delay `0` and wind/gust `0` before formal checkpoint validation.
- S1 feedforward PPO trained from scratch with `256` envs, `300` iterations, seed `42`; no resume flag and no S0 checkpoint load.
- Final local checkpoint, not tracked by Git:

```text
logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_20-57-39_m7b_s1_ff_300_seed42/model_299.pt
```

- Final checkpoint SHA-256: `2e159913068a795eef6b48924267b453876f547d6113cffc3c69efaac61de3dc`.
- Formal validation passed on validation split, seed `4242`, `64` envs, `8` episodes/env, `512` total episodes, exactly `128` episodes per target mode.
- Overall validation metrics: `success_rate=1.0`, `collision_risk_rate=0.0`, `workspace_violation_rate=0.0`, `height_violation_rate=0.0`, `speed_violation_rate=0.0`, successful offset p95 `0.3154950439929962 m`, successful relative-speed p95 `0.16483217477798462 m/s`, deterministic inference `max_abs_delta=0.0`.
- Per-mode success rates: ConstantAcceleration `1.0`, ConstantTurn `1.0`, ConstantVelocity `1.0`, PiecewiseAcceleration `1.0`.
- Stage-isolation diagnostics from formal validation: tau min/max `0.7646484375/1.44140625`, acceleration min/max `0.80859375/1.1996093988`, speed min/max `0.9001953006/1.0997558832`, drag min/max `0.0002288819/0.1494140625`, action delay min/max `0/0`, steady/current wind max `0.0/0.0`, gust target/current max `0.0/0.0`, Actor `25D`, Critic `65D`, finite diagnostics `true`.
- All pre-registered S1 gates passed.
- Stop state: do not execute M7B-S2, M7B-S3, M7B-S4, GRU training, M7C, commit, tag, or push without explicit user confirmation.
