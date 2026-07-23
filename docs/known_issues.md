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

## M5 Accepted Non-Blocking Issues

M5 was accepted with non-blocking issues. These items are technical debt to address before or during later milestone work, especially before M6 changes done-mask or hidden-state behavior.

1. `tests/test_m5_rewards.py` has some expected values that directly call the production reward function instead of a fully independent mathematical reference implementation. Current risk is low because pure unit tests, oracle audit, training, and 256-episode independent validation all passed. If reward formulas are changed later, add independent reference tests.

2. The 5-iteration PPO startup test metrics are noisy. That test verifies training linkage, finite loss behavior, and checkpoint creation; it is not a convergence criterion. The 300-iteration from-scratch training run and independent validation passed.

3. `Episode_Termination/time_out` logs an absolute episode count rather than a proportion. This follows the current Isaac Lab extras logging convention, but readers must interpret it as a count.

4. Success and collision one-time events do not yet have dedicated multi-step unit tests. Runtime audits verified event accounting and reset behavior. Add this test coverage before M6 modifies done masks or recurrent hidden-state reset handling.

## M6 Recurrent Performance Limitation

M6 verified the following capabilities:

- GRU recurrent PPO chain.
- Independent Actor and Critic memory.
- Done-mask hidden reset.
- Checkpoint save, load, and resume.
- History sensitivity.
- Mixed-mode safe operation.

However, the fair feedforward baseline outperformed GRU on the current task:

- Success offset p95: feedforward about `0.196 m`, GRU about `0.463 m`.
- Relative speed p95: feedforward about `0.109 m/s`, GRU about `0.222 m/s`.
- Return: feedforward about `2192`, GRU about `1986`.
- Convergence time: feedforward about `3.23 s`, GRU about `5.38 s`.

Therefore M6 proves that GRU can retain and use history, but it does not prove that implicit history-based prediction provides a task performance advantage.

Likely reasons include:

- The current 25D Actor observation is already close to Markov state.
- Current target relative velocity is sufficient for the existing target-motion range.
- Target maneuver intensity is not high enough to create strong partial observability.
- GRU optimization difficulty and the fixed training budget offset the potential benefit of historical information.

History sensitivity must not be described as proof of a performance advantage.

M6 non-blocking issues:

1. Some configuration tests are string-based and do not fully instantiate configuration classes.
2. History sensitivity uses synthetic observations and verifies mechanism only.
3. Per-mode cycling associates environment ID with mode, but statistical coverage is sufficient.
4. One Isaac Sim startup-stage segmentation fault occurred and was not reproduced in subsequent formal runs.
