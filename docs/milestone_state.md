# Milestone State

Current milestone: M6
Status: accepted_with_major_limitation
Last completed milestone: M6
M4 accepted tag: m4-accepted
M4 accepted commit: 36592b6a14cd1a00d6bb689b3a33d27fe610a3b1
M5 implementation commit: 887bb20a3d5a44eac479fc451fab89aa18296b57
M5 accepted tag: m5-accepted
M5 accepted commit: 61e3a8107b966bf146b46a3855b0ac256cdf53c2
M5 independent audit result: ACCEPT M5 WITH NON-BLOCKING ISSUES
M6 implementation commit: 2f4dd9c85b931075294f59bafe7e39d9b2127765
M6 independent audit result: ACCEPT M6 WITH MAJOR LIMITATION
Next milestone: M7, not authorized

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

M7 has not started.

## Authorized Work

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
- M7 is not authorized.

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

M6 is accepted with major limitation. M7 is not authorized and has not started.

## Next Milestone Guard

- M5 has passed user acceptance.
- M6 has passed user acceptance with major limitation.
- Do not enter M7 without explicit user confirmation.
