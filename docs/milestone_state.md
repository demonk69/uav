# Milestone State

Current milestone: M2
Status: completed_pending_user_acceptance
Last completed milestone: M1
Next milestone: M3, not authorized

## Authorized Work

- 每环境创建ego和target两个简化占位实体
- 真值位置和速度状态
- target匀速、固定高度运动
- ego固定或静止
- p_rel_w和v_rel_w计算
- 每环境独立随机化
- 固定seed复现
- reset测试
- 匀速解析解测试
- 16环境10000步稳定性测试

## Forbidden In M2

- ConstantAcceleration、ConstantTurn、PiecewiseAcceleration
- 通用TargetMotionGenerator库
- 确定性偏置交会基线
- 复杂奖励设计
- PPO训练
- GRU或LSTM
- 非对称Actor-Critic
- B样条或未来轨迹输出
- Crazyflie
- Multirotor/Thruster
- Pegasus
- PX4
- ROS 2
- 相机或第一阶段感知网络

## Notes

- M2 started after user acceptance of M1.
- M2 implemented dual `ego` and `target` placeholder entities, truth-state tensors, fixed-height constant-velocity target motion, stationary ego state, `[p_rel_w, v_rel_w]` observations, reset randomization, reproducibility tests, and script diagnostics.
- M2 task tensors are the single source of truth. The `RigidObject` placeholders are synchronized state and visualization carriers only, not final UAV dynamics.
- M2 validation passed on 2026-07-21:
- M2 closure audit passed on 2026-07-22:
- syntax check: passed for 20 Python files
- pytest: `10 passed in 0.76s`
- runtime audit: `scripts/audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42 --device cuda:0 --headless`, passed
- runtime audit metrics: `sim_dt=0.01`, `decimation=2`, `env_step_dt=0.02`, `target_actual_average_speed=0.675286054611206`, `max_analytic_position_error=0.000335693359375`, `target_asset_position_sync_error=0.0`, `target_asset_velocity_sync_error=0.0`, `ego_max_drift=0.0`, `min_relative_distance=4.625568866729736`, `finite=true`
- partial reset audit: envs `[1, 3, 7]`, elapsed reset to zero, max untouched-state delta `0.0`, selected position sync error `4.76837158203125e-07`
- seed audit: same-seed reseed passed, different seed changed state, 16 envs independently randomized
- independent process seed probes: seed `42` digest `42721797c56c73943f8f1f577b837aaedf4b6481db5e33348a715aecccf1c6e4` reproduced twice; seed `43` digest `cd18fea5d4dba4440000cd9cf14e679ca8e759f850f99e821b22f7a15aae4a1e` differed
- zero-agent 10000 steps, 16 envs, `cuda:0`, passed with finite truth state, target moved, ego static
- random-agent 10000 steps, 16 envs, `cuda:0`, passed with finite truth state, target moved, ego static
- Spawned sphere placeholders use gravity-disabled dynamic `RigidObject`s, not `kinematic_enabled=True`, because kinematic spawned spheres caused PhysX GPU illegal-memory-access errors during startup on this host. Collision remains explicitly enabled in the asset schema; M2 runtime audit verifies no contact by center distance and no physics-induced disturbance by asset synchronization and ego drift checks.
- No M3 work is authorized.
- M2 closure commit and `review/m2` branch push were authorized by the user after audit pass.
