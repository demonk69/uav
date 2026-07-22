# Milestone State

Current milestone: M5
Status: implemented_and_locally_verified_pending_user_acceptance
Last completed milestone: M4
M4 accepted tag: m4-accepted
M4 accepted commit: 36592b6a14cd1a00d6bb689b3a33d27fe610a3b1
Next milestone: M6, not authorized

## Authorized Work

- 独立RL任务
- 3维短期目标速度动作
- 简化加速度受限ego动力学
- 25维可部署Actor观测
- 57维特权Critic观测
- 非对称前馈Actor-Critic
- 奖励与终止条件
- 随机动作稳定性审计
- 前馈PPO短训练
- checkpoint生成
- deterministic play/evaluation

## Forbidden In M5

- GRU
- LSTM
- recurrent PPO
- 显式目标未来轨迹
- B样条轨迹输出
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
- The original `Isaac-Uav-Rendezvous-Direct-v0` task must remain an M2/M3 regression task with stationary ego and no-op action.
- The original `Isaac-Uav-Rendezvous-Baseline-v0` task must remain the M4 deterministic baseline and must not be affected by RL actions.
- M6 is not authorized.

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

## Pending User Decision

- M5 is ready for user review/acceptance.
- Do not enter M6 or create a Git commit without explicit user confirmation.
