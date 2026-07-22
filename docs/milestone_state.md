# Milestone State

Current milestone: M3
Status: completed_pending_user_acceptance
Last completed milestone: M2
M2 acceptance commit: 34709da0c07c60f1aa8e1bf34ecf476992d54e46
Next milestone: M4, not authorized

## Authorized Work

- 通用TargetMotionGenerator接口
- ConstantVelocity
- ConstantAcceleration
- ConstantTurn
- PiecewiseAcceleration
- 每环境独立参数
- 固定seed复现
- train/validation/test参数分离
- M3实际环境运行审计
- 继续使用M2的简化占位实体

## Forbidden In M3

- ego交会控制器
- 确定性基线
- v_cmd_w动作控制
- 奖励函数
- PPO训练
- GRU或LSTM
- 非对称Actor-Critic
- B样条轨迹
- Crazyflie
- Multirotor/Thruster
- Pegasus
- PX4
- ROS 2
- 相机或第一阶段感知网络

## Notes

- M2 accepted by user; acceptance commit is `34709da0c07c60f1aa8e1bf34ecf476992d54e46`.
- Local `review/m2` was fast-forward merged into `main`; annotated tag `m2-accepted` was created at the M2 acceptance commit.
- Remote `main` and tag `m2-accepted` were pushed before starting M3 work.
- M3 work is confined to branch `feature/m3` until completion.
- M4 is not authorized.

## M3 Implementation Summary

- Added vectorized `uav_rendezvous_rl.motions` target motion library.
- Implemented common `TargetMotionGenerator` protocol and `MotionState` state container.
- Implemented `ConstantVelocity`, `ConstantAcceleration`, `ConstantTurn`, and `PiecewiseAcceleration` generators.
- Added `TargetMotionManager` with per-env mixed modes, seeded reset, partial reset, split-aware sampling, validity checks, and diagnostics.
- Integrated M3 target motion manager into the DirectRLEnv while keeping Actor observations limited to `[p_rel_w, v_rel_w]`.
- Kept M2 constant-velocity helper as a compatibility wrapper over the M3 constant-velocity formula.
- Updated M2 runtime audit to force ConstantVelocity so M2 regression remains analytically checkable under M3 defaults.
- Added `scripts/audit_m3_motion_runtime.py` for runtime validation of all M3 motion modes, independent analytic references, Actor isolation, task tensor truth authority, asset sync, seed/split reproducibility, partial reset, no contact, and finite state.

## M3 Verification

- Syntax check: `syntax ok: 38 files`.
- Pytest: `27 passed in 0.83s`.
- M2 regression runtime audit passed: `scripts/audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42 --device cuda:0 --headless`.
- M3 runtime audit passed: `scripts/audit_m3_motion_runtime.py --num_envs 16 --steps 5000 --seed 42 --split train --device cuda:0 --headless`.
- M3 audit mode coverage: ConstantVelocity 4, ConstantAcceleration 4, ConstantTurn 4, PiecewiseAcceleration 4.
- M3 independent analytic overall max errors: position 3.5762786865234375e-06, velocity 2.9802322387695312e-08, acceleration 9.02347911668766e-10.
- M3 audit safety: all-physics-substep minimum relative distance 4.436470985412598 m, `d_safe` 0.75 m, no collision risk.
- M3 audit validity: finite truth state, no invalid target motion, no ego drift, asset sync within tolerance.
- M3 audit max target speed observed: 1.7801588773727417 m/s.
- M3 audit max target acceleration observed: 0.03640248253941536 m/s^2.
- M3 audit workspace max absolute coordinate observed: 93.41694641113281 m.
- M3 audit PiecewiseAcceleration segment switches: total 562, per PiecewiseAcceleration env [136, 136, 143, 147], every PiecewiseAcceleration env switched at least once.
- M3 seed/split digest audit:
- train seed 42 process A digest `9a8feaf028dda96abe174630ec00d78b50eb565ba0b54febb07cb965b06609e7`.
- train seed 42 process B digest `9a8feaf028dda96abe174630ec00d78b50eb565ba0b54febb07cb965b06609e7`.
- train seed 43 digest `17f36606ab398a1ebef2e3f14f088a634501615300b06f6c146415cc08630da9`.
- validation seed 42 digest `ed8c437ab2ce3d97f05ade72086c0e15687f3d79aa22aa7f88b50eb4984c15f2`.
- test seed 42 digest `cd8d9b9f47d6d05d63fd9280b5ce8fb717fdbf4d51cc813118cc3beede8feadf`.
- M3 split config ranges are distinct across train, validation, and test, with unique seed offsets.
- M3 Actor observation check: only `policy` group, shape `[16, 6]`, exactly `[p_rel_w, v_rel_w]`, no mode id, motion parameters, future schedule, or future target state.
- Zero-agent 10000-step smoke passed on `cuda:0` headless.
- Random-agent 10000-step smoke passed on `cuda:0` headless.

## Current Review State

- Branch: `feature/m3`.
- Review branch target: `review/m3`.
- Do not merge or enter M4 without explicit user confirmation.
