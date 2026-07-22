# Known Issues

## M2 Placeholder Rigid Objects

On the current host, spawned sphere placeholders with `kinematic_enabled=True` triggered PhysX GPU `illegal memory access` errors during Isaac Lab startup/reset. M2 therefore uses gravity-disabled dynamic `RigidObject` sphere placeholders.

The M2 placeholders are only synchronized state and visualization carriers. They are not the final UAV dynamics model. The authoritative M2 truth state is the task tensor state maintained by `UavRendezvousEnv`, including `p_ego_w`, `v_ego_w`, `p_target_w`, and `v_target_w`; observations and termination checks read those tensors, and runtime audit checks that the placeholder `RigidObject` data stays synchronized to them.

Placeholder collision remains explicitly enabled in the asset schema because this is the stable configuration verified on this host. M2 remains non-contact: runtime audit checks that the center distance stays above `d_safe`, that synchronized asset positions and velocities match task tensors, and that ego has no physics-induced drift.

Do not retry `kinematic_enabled=True` during M2 closure. Do not modify Isaac Lab, Isaac Sim, Pegasus, system Python dependencies, or NVIDIA driver components for this issue.

## M3 TargetMotionManager CUDA Synchronization Debt

`TargetMotionManager` currently has several high-frequency CUDA-path calls such as `torch.any(...).item()` and `count_nonzero(...).item()`. These can cause CPU-GPU synchronization. M3 functionality is correct and accepted, but before M5 large-scale parallel training this path must be profiled and non-essential synchronization should be reduced.

Do not refactor the manager as part of M3 acceptance archival work.

## M4 Baseline Scope Limitation

M4 uses exact target truth, simplified ego dynamics, and current-state short-horizon extrapolation. The low audited errors for ConstantVelocity targets show that the deterministic offset rendezvous baseline and task plumbing are correct under these assumptions.

These M4 results do not represent final policy performance under perception noise, observation delay, external disturbances, aggressive or complex target maneuvers, or real multirotor dynamics. Those conditions remain outside M4 and must not be inferred from the deterministic baseline audit.

## M4 Baseline Initial Geometry Resampling Debt

`BaselineInitialGeometryCfg.max_resample_attempts` is currently not wired into reset sampling. The fixed M4 initial geometry ranges naturally satisfy the safety conditions, so this does not affect M4 acceptance.

Before M5 introduces randomized `b_des_w` and wider initial bearings, reset must implement finite-attempt, vectorized batch resampling. Infinite `while` loops are forbidden.
