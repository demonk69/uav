# M7B Verification

Status: implementation gates M7B-G1 through M7B-G5 complete. M7B-S1 dynamics-only training and validation complete. M7B-S2 through M7B-S4 are not started.

Independent G5 audit: `docs/m7b_g5_independent_audit.md` concluded `ACCEPT M7B-G5 WITH NON-BLOCKING ISSUES`. The audited S0 checkpoint SHA-256 is `7254b343972ab30a27c48dcb7fd5c5a90079ef6be25c067ec92cc0679f32e5f8`.

Session start for S1 execution:

- Role: `EXECUTOR_A`.
- Initial branch: `feature/m7b`.
- Initial HEAD: `555daeb598ef633f7e2dbf8b86a12148cd48cf34`.
- Initial status: M7B implementation worktree changes present, including `docs/m7b_g5_independent_audit.md`; no commit, tag, or push requested.

## G1: Syntax And Pure Tests

- `git diff --check` passed.
- Isaac compileall passed for changed source, scripts, and tests.
- Full pure pytest passed: `134 passed in 1.85s`.

## G2: Nominal Dynamics Equivalence

- M7B dynamics, parameter sampling, and wind unit tests passed: `12 passed`.
- Nominal M7B dynamics match the accepted M5-M7A single-step formula within `1e-6`.
- Position integration uses `p + v*dt + 0.5*a*dt^2`.

## G3: Runtime Contracts

- M7B feedforward S0 runtime audit passed.
- M7B GRU S0 runtime audit passed on retry after an Isaac startup segfault occurred before project environment initialization.
- Verified Actor observation `25D`, Critic observation `65D`, action dimension `3`, clean S0 observation pipeline, FIFO semantics, and secondary GRU critic input `65D`.
- Segfault fact: the first M7B GRU S0 runtime-audit attempt exited during Isaac startup before project environment initialization. The retry passed. No Isaac Lab, Isaac Sim, system dependency, or driver files were modified.

## G4: Runtime And Regression Audits

- M7B-S4 combined 10000-step runtime audit passed.
- Compact M2, M3, M4, M5, and M6 regression audits passed.
- Compact M7A GRU observation-pipeline regression passed on retry after an Isaac startup segfault occurred before project environment initialization.

Exact M7A compact GRU retry command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/audit_m7_observation_pipeline.py --task Isaac-Uav-Rendezvous-M7A-GRU-v0 --m7a_stage 4 --num_envs 4 --steps 8 --seed 43 --device cuda:0 --headless
```

M7A compact GRU retry result: passed with `policy_obs_dim=25`, `critic_obs_dim=57`, finite check `true`, no-future-leakage check present, hidden reset check present, partial reset check present, and `done_count=0` over 8 audit steps. The previous seed `42` attempt exited during Isaac startup before the project audit report was produced.

## G5: Clean Startup And Validation

Training command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 0 --num_envs 256 --max_iterations 300 --seed 42 --run_name m7b_s0_ff_300_seed42 --device cuda:0 --headless
```

Checkpoint:

```text
logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_01-30-42_m7b_s0_ff_300_seed42/model_299.pt
```

Checkpoint SHA-256:

```text
7254b343972ab30a27c48dcb7fd5c5a90079ef6be25c067ec92cc0679f32e5f8
```

Validation command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/evaluate.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 0 --policy trained --checkpoint /home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_01-30-42_m7b_s0_ff_300_seed42/model_299.pt --num_envs 64 --episodes 8 --seed 4242 --split validation --target_motion_mode Mixed --force_mode_cycle_on_reset --determinism_check --device cuda:0 --headless
```

Validation result:

- Episodes: `512`, balanced `128` per target mode.
- Success rate: `1.0`.
- Collision risk rate: `0.0`.
- Workspace violation rate: `0.0`.
- Height violation rate: `0.0`.
- Speed violation rate: `0.0`.
- Successful offset error p95: `0.3143340051 m`.
- Successful relative speed p95: `0.1617015600 m/s`.
- Deterministic inference repeat: `max_abs_delta=0.0`.
- Actor observation dimension: `25`.
- Critic observation dimension: `65`.
- Finite diagnostics: `true`.
- Clean M7B-S0 settings verified: tau, acceleration, and speed scales `1.0`; drag `0.0`; delay `0`; wind/gust `0`.

## S1 Locked Protocol

This protocol is locked before seeing M7B-S1 training or validation results.

- Task: `Isaac-Uav-Rendezvous-M7B-Feedforward-v0`.
- Policy: feedforward PPO primary policy only; no GRU training in this run.
- Stage: `--m7b_stage 1`, dynamics randomization only.
- Observation degradation: disabled; clean M7A pipeline only.
- Action delay: must remain `0`.
- Wind/gust: must remain `0`.
- Actor observation: `25D`.
- Critic observation: `65D`.
- Training: from scratch, not resumed from S0; `256` envs, `300` iterations, seed `42`.
- Validation: validation split, seed `4242`, `64` envs, `8` episodes/env, `512` total episodes, balanced `128` per target mode, determinism check enabled.

Pre-registered S1 gates:

- `collision_risk_rate=0`.
- `workspace_violation_rate=0`.
- `height_violation_rate=0`.
- `speed_violation_rate=0`.
- `finite diagnostics=true`.
- deterministic `max_abs_delta=0`.
- overall `success_rate >= 0.95`.
- each target-mode `success_rate >= 0.90`.
- successful offset p95 `<= 0.50 m`.
- successful relative-speed p95 `<= 0.30 m/s`.

## S1 Preflight

- `git diff --check` passed.
- Isaac compileall passed:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p -m compileall scripts source/uav_rendezvous_rl/uav_rendezvous_rl tests
```

- Targeted M7B tests passed after updating the smoke-test assertion to the corrected M7B control-plane wording: `41 passed in 0.92s`.

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p -m pytest tests/test_m7b_*.py tests/test_env_smoke.py -q
```

## S1 Runtime Audits

Short startup audit command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/audit_m7b_dynamics.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 1 --num_envs 4 --steps 8 --seed 42 --policy zero --device cuda:0 --headless
```

Short startup audit result:

- Passed with Actor `25D`, Critic `65D`, finite diagnostics `true`.
- S1 dynamics randomization active: tau min/max `0.8014526367/1.3974609375`, acceleration min/max `0.9281250238/1.1156250238`, speed min/max `1.0343749523/1.0648437738`, drag min/max `0.0011718750/0.0963867232`.
- S2/S3 disturbances disabled: action delay min/max `0/0`, current wind norm max `0.0`, gust target/current norm max `0.0/0.0`.

10000-step runtime audit command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/audit_m7b_dynamics.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 1 --num_envs 16 --steps 10000 --seed 42 --policy random --target_motion_mode Mixed --device cuda:0 --headless
```

10000-step runtime audit result:

- First attempt exited during Isaac startup with `appState='startup'`, `UptimeSeconds='0'`, before project environment initialization.
- Retried the exact same command with the same seed and arguments; retry passed.
- Retry diagnostics: Actor `25D`, Critic `65D`, finite diagnostics `true`, total steps `10000`, collision risk count `0`, workspace violation count `0`, height violation count `0`.
- S1 dynamics randomization active: tau min/max `0.7880859375/1.46337890625`, acceleration min/max `0.8335937858/1.1851562262`, speed min/max `0.9152343273/1.0960693359`, drag min/max `0.0023437501/0.1395996213`.
- S2/S3 disturbances disabled: action delay min/max `0/0`, current wind norm max `0.0`, gust target/current norm max `0.0/0.0`.

Validation-split stage-isolation command before formal checkpoint validation:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/evaluate.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 1 --policy zero --num_envs 64 --episodes 1 --seed 4242 --split validation --target_motion_mode Mixed --force_mode_cycle_on_reset --device cuda:0 --headless
```

Validation-split stage-isolation result:

- Target split `validation`, exactly `16` envs per target mode at reset.
- Dynamics randomization covered the S1 validation ranges in the sampled batch: tau min/max `0.7738037109/1.494140625`, acceleration min/max `0.8000004292/1.1953125`, speed min/max `0.9039062262/1.099609375`, drag min/max `0.0000001500/0.1480957121`.
- S2/S3 disturbances disabled: action delay min/max `0/0`, current wind norm max `0.0`, gust target/current norm max `0.0/0.0`.
- Actor `25D`, Critic `65D`, finite diagnostics `true`, clean observation pipeline confirmed.

## S1 Training

Training command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 1 --num_envs 256 --max_iterations 300 --seed 42 --run_name m7b_s1_ff_300_seed42 --device cuda:0 --headless
```

Run directory:

```text
logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_20-57-39_m7b_s1_ff_300_seed42
```

Final checkpoint:

```text
logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_20-57-39_m7b_s1_ff_300_seed42/model_299.pt
```

Final checkpoint SHA-256:

```text
2e159913068a795eef6b48924267b453876f547d6113cffc3c69efaac61de3dc
```

Training result:

- Completed the locked from-scratch `256` env, `300` iteration, seed `42` run.
- No resume flag and no checkpoint load were used.
- Final logged training iteration reported `success_rate=1.0000` and zero collision-risk, workspace, height, speed, attitude, NaN/Inf, and target-motion-invalid terminations.

## S1 Formal Validation

Validation command:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME /home/lab_726/IsaacLab/isaaclab.sh -p scripts/evaluate.py --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 1 --policy trained --checkpoint /home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_20-57-39_m7b_s1_ff_300_seed42/model_299.pt --num_envs 64 --episodes 8 --seed 4242 --split validation --target_motion_mode Mixed --force_mode_cycle_on_reset --determinism_check --device cuda:0 --headless
```

Validation result:

- Episodes: `512`, balanced `128` per target mode.
- Overall success rate: `1.0`.
- Collision risk rate: `0.0`.
- Workspace violation rate: `0.0`.
- Height violation rate: `0.0`.
- Speed violation rate: `0.0`.
- Deterministic inference repeat: `max_abs_delta=0.0`.
- Successful offset error p95: `0.3154950439929962 m`.
- Successful relative speed p95: `0.16483217477798462 m/s`.
- Average return: `2119.27783203125`.
- Convergence time p95: `5.199999809265137 s`.

Per-mode validation result:

| Mode | Episodes | Success rate | Offset p95 (m) | Relative-speed p95 (m/s) |
| --- | ---: | ---: | ---: | ---: |
| ConstantAcceleration | 128 | 1.0 | 0.31992992758750916 | 0.16503219306468964 |
| ConstantTurn | 128 | 1.0 | 0.31152456998825073 | 0.16426779329776764 |
| ConstantVelocity | 128 | 1.0 | 0.3157779276371002 | 0.1640460193157196 |
| PiecewiseAcceleration | 128 | 1.0 | 0.3155469000339508 | 0.16569751501083374 |

Stage-isolation diagnostics from formal validation:

- Actor observation dimension: `25`.
- Critic observation dimension: `65`.
- Finite diagnostics: `true`.
- Clean observation pipeline: delay `0`, dropout `0.0`, noise `0.0`, position/velocity update period `1`.
- Dynamics randomization active: tau min/max `0.7646484375/1.44140625`, acceleration min/max `0.80859375/1.1996093988`, speed min/max `0.9001953006/1.0997558832`, drag min/max `0.0002288819/0.1494140625`.
- S2/S3 disturbances disabled: action delay min/max `0/0`, steady wind norm max `0.0`, current wind norm max `0.0`, gust target/current norm max `0.0/0.0`.

Gate verdict:

- `collision_risk_rate=0`: PASS.
- `workspace_violation_rate=0`: PASS.
- `height_violation_rate=0`: PASS.
- `speed_violation_rate=0`: PASS.
- `finite diagnostics=true`: PASS.
- deterministic `max_abs_delta=0`: PASS.
- overall `success_rate >= 0.95`: PASS.
- each target-mode `success_rate >= 0.90`: PASS.
- successful offset p95 `<= 0.50 m`: PASS.
- successful relative-speed p95 `<= 0.30 m/s`: PASS.

## Stop State

- M7B implementation and clean S0 gate are verified.
- M7B-S1 dynamics-only training and validation are complete and passed the pre-registered S1 gates.
- M7B-S2, M7B-S3, and M7B-S4 formal robustness training and validation are not started.
- M7C is not authorized.
- No commit or push was created.
