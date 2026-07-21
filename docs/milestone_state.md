# Milestone State

Current milestone: M1
Status: completed_pending_user_acceptance
Last completed milestone: M0
Next milestone: M2, not authorized

## Authorized Work

- 最小外部DirectRLEnv项目
- editable安装
- Gymnasium任务注册
- 16环境reset/step
- zero/random agent
- 10000步稳定性测试
- pytest

## Forbidden In M1

- 双无人机完整真值环境
- 目标运动库
- 基线控制器
- 奖励设计
- PPO训练
- GRU
- 非对称Actor-Critic
- B样条轨迹
- Pegasus
- PX4
- ROS 2
- Crazyflie
- Multirotor/Thruster

## Notes

- M1 started after M0 documentation was created.
- Created `AGENTS.md` and initialized this milestone state file.
- Corrected the empty Action Space V0 code block in `docs/implementation_plan.md`.
- Created the minimal external DirectRLEnv package, scripts, and tests authorized for M1.
- Editable install passed with Isaac Lab Kit Python.
- Gymnasium registration passed for `Isaac-Uav-Rendezvous-Direct-v0`.
- Pytest passed: 3 tests.
- `zero_agent.py` completed 10000 steps with 16 cuda:0 environments.
- `random_agent.py` completed 10000 steps with 16 cuda:0 environments.
- No NaN, Inf, CUDA error, or crash was observed in M1 smoke scripts.
- Isaac Sim emitted startup/shutdown warnings; none blocked M1 acceptance.
- No M2 work is authorized.
- No Git commit is authorized.
