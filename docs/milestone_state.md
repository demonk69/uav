# Milestone State

Current milestone: M4
Status: accepted
Last completed milestone: M4
M4 implementation commit: 302b6b6a63b8de7b2268c0077b17d6eed3041d91
M4 final acceptance commit: cba8cafe84c7ca0f21e22c9b571d313d9a0fc855
Next milestone: M5, authorized but not started

## Authorized Work

- 确定性非学习偏置交会基线
- 目标当前状态的短期匀速外推
- 简化ego速度和加速度受限动力学
- 独立Baseline任务
- 偏置误差、相对速度、安全性和成功率统计
- ConstantVelocity正式验收
- 其他运动模式压力测试

## Forbidden In M4

- PPO训练
- GRU/LSTM
- Actor/Critic网络
- RL奖励训练
- B样条轨迹
- Crazyflie
- Multirotor/Thruster
- Pegasus
- PX4
- ROS 2
- 感知网络
- 读取目标模式标签
- 读取运动生成参数
- 读取目标未来状态或未来schedule

## Notes

- M2 accepted by user; acceptance tag is `m2-accepted`.
- M3 accepted by user; acceptance tag is `m3-accepted` at commit `605325a9142aa534d20b2d52ea6533cf598c2c12`.
- M3 implementation commit is `17e05e8ebed2bbc100dc2eef9d8b0fe4486846c5`.
- M4 starts from `m3-accepted` on branch `feature/m4`.
- The original `Isaac-Uav-Rendezvous-Direct-v0` task must remain an M2/M3 regression task with stationary ego and no-op action.
- M4 passed user technical acceptance after collision risk accounting was fixed.
- M5 is authorized but not started. No M5 functionality has been implemented.

## M4 Implementation Summary

- Added pure PyTorch deterministic baseline controller utilities under `uav_rendezvous_rl.controllers`.
- Implemented current-state ConstantVelocity target extrapolation: `p_target_pred_w = p_target_w + v_target_w * T_pred`.
- Implemented offset goal command: `p_goal_w = p_target_pred_w + b_des_w` and `v_cmd_w = (p_goal_w - p_ego_w) / T_pred`.
- Implemented acceleration-limited ego kinematics with vector-norm velocity, acceleration, and absolute speed limits.
- Added independent `Isaac-Uav-Rendezvous-Baseline-v0` task without modifying the original Direct task behavior.
- Added M4 reset geometry that samples `p_ego_initial_w = p_target_initial_w + b_des_w + delta_initial_w` with a non-contact path to the offset point.
- Added M4 episode metrics and diagnostics for offset error, relative speed, success hold, collision risk, workspace, speed limit, finite state, saturation, and asset sync.
- Added `scripts/audit_m4_baseline_runtime.py` for nominal fixed ConstantVelocity, random ConstantVelocity, and stress-mode runtime audits.
- Added unit tests for prediction, control law, limits, kinematics, initial geometry, causality, and ConstantVelocity convergence.

## M4 Verification

- Syntax check passed: `env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m compileall -q scripts source tests`.
- Pytest passed: `41 passed in 0.97s`.
- Small M4 smoke passed: `scripts/audit_m4_baseline_runtime.py --num_envs 8 --episodes 1 --seed 42 --split train --device cuda:0 --headless`.
- M2 regression runtime audit passed: `scripts/audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42 --device cuda:0 --headless`.
- M3 regression runtime audit passed: `scripts/audit_m3_motion_runtime.py --num_envs 16 --steps 5000 --seed 42 --split train --device cuda:0 --headless`.
- Formal M4 baseline audit passed: `scripts/audit_m4_baseline_runtime.py --num_envs 64 --episodes 5 --seed 42 --split train --device cuda:0 --headless`.
- M4 nominal fixed ConstantVelocity: success rate `1.0`, collision count `0`, success offset error p95 `7.62939453125e-06`, success relative speed p95 `8.58306884765625e-06`.
- M4 random ConstantVelocity: success rate `1.0`, collision count `0`, success offset error p95 `4.206398443784565e-05`, success relative speed p95 `8.18166954559274e-05`.
- M4 stress ConstantAcceleration: success rate `1.0`, collision count `0`, success offset error p95 `0.0022186809219419956`, success relative speed p95 `7.15116475475952e-05`.
- M4 stress ConstantTurn: success rate `1.0`, collision count `0`, success offset error p95 `0.009585591964423656`, success relative speed p95 `0.0008776930626481771`.
- M4 stress PiecewiseAcceleration: success rate `1.0`, collision count `0`, success offset error p95 `0.00256736995652318`, success relative speed p95 `0.004727173130959272`.
- Direct zero-agent 10000-step smoke passed on `Isaac-Uav-Rendezvous-Direct-v0`.
- Direct random-agent 10000-step smoke passed on `Isaac-Uav-Rendezvous-Direct-v0`.

## Current Review State

- Branch: `feature/m4`.
- Review branch target: `review/m4`.
- M4 implementation commit: `302b6b6a63b8de7b2268c0077b17d6eed3041d91`.
- M4 final acceptance commit: `cba8cafe84c7ca0f21e22c9b571d313d9a0fc855`.
- Archive commit message: `Docs: finalize M4 acceptance`.
