# M5 Verification

Date: 2026-07-22

Status: implemented and locally verified, pending user acceptance.

## Scope

M5 adds the independent feedforward RL task `Isaac-Uav-Rendezvous-RL-v0` while preserving the M2/M3 Direct task and M4 Baseline task.

Key constraints verified:

- Actor uses only deployable current observations.
- Actor observation dimension is 25.
- Critic observation dimension is 57 under the separate `critic` group.
- PPO policy is feedforward `ActorCritic`; no GRU, LSTM, or recurrent PPO is used.
- Action space is raw unbounded 3D action with environment-owned `v_cmd_w = v_max * tanh(a_raw)` mapping.
- RSL-RL wrapper uses `clip_actions=None`.

## Code Fixes During Final Verification

- Added episode-history fields for success-event metrics: `success_offset_error` and `success_relative_speed`.
- Updated evaluation summaries to report success-event offset/speed instead of final episode-end values for successful episodes.
- Fixed reward accounting so `success_completion_bonus` is paid only on the policy step where the success hold first completes. The per-step success-hold bonus remains unchanged.

## Verification Commands

All Isaac Lab commands were run through `/home/lab_726/IsaacLab/isaaclab.sh` with `PYTHONPATH`, `PYTHONHOME`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, and `VIRTUAL_ENV` cleared.

Static and unit checks:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m py_compile source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_rl_env.py scripts/evaluate.py
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pytest tests -q
```

Result: `55 passed`.

M5 runtime audit after the reward fix:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p scripts/audit_m5_rl_runtime.py --num_envs 64 --steps 10000 --seed 42 --split train --scenario all --device cuda:0 --headless
```

Result: audit completed for `zero_constant_velocity`, `random_constant_velocity`, `random_mixed_modes`, and `oracle_constant_velocity`. All scenarios remained finite. The oracle path achieved `success_rate=1.0` with zero collision-risk, workspace, height, and speed-limit counts.

## Training

Final corrected training command:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Uav-Rendezvous-RL-v0 --num_envs 256 --max_iterations 300 --seed 42 --device cuda:0 --headless --run_name m5_rewardfix_300_seed42
```

Final checkpoint:

```text
/home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_19-04-26_m5_rewardfix_300_seed42/model_299.pt
```

Training result at iteration 299:

| Metric | Value |
| --- | ---: |
| `Metrics/success_rate` | `1.0000` |
| `Metrics/final_offset_error` | `0.0885` |
| `Episode_Termination/collision_risk` | `0.0000` |
| `Episode_Termination/workspace_violation` | `0.0000` |
| `Episode_Termination/height_violation` | `0.0000` |
| `Episode_Termination/speed_violation` | `0.0000` |

## Deterministic Validation

Validation protocol:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p scripts/evaluate.py --task Isaac-Uav-Rendezvous-RL-v0 --policy trained --num_envs 64 --episodes 4 --seed 4242 --split validation --checkpoint /home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_19-04-26_m5_rewardfix_300_seed42/model_299.pt --device cuda:0 --headless
```

Result summary for 256 completed episodes:

| Metric | Value |
| --- | ---: |
| `success_rate` | `1.0` |
| `collision_risk_rate` | `0.0` |
| `workspace_violation_rate` | `0.0` |
| `height_violation_rate` | `0.0` |
| `speed_violation_rate` | `0.0` |
| `success_offset_error.p95` | `0.3592815101146698` |
| `success_relative_speed.p95` | `0.16953714191913605` |
| `final_offset_error.mean` | `0.09622277319431305` |
| `final_offset_error.p95` | `0.1540052890777588` |
| `final_relative_speed.p95` | `0.0019355688709765673` |
| `convergence_time.mean` | `4.09765625` |
| `convergence_time.p95` | `5.579999923706055` |
| `action_saturation_fraction.mean` | `0.0008563250303268433` |
| `acceleration_saturation_fraction.mean` | `0.045303113758563995` |

Acceptance thresholds checked:

| Requirement | Result |
| --- | --- |
| Validation success rate `>= 80%` | pass, `100%` |
| Collision-risk rate `= 0` | pass, `0` |
| Successful-episode offset p95 `< 0.50 m` | pass, `0.3593 m` |
| Successful-episode relative-speed p95 `< 0.30 m/s` | pass, `0.1695 m/s` |

## Policy Comparison

All comparison runs used `--num_envs 64 --episodes 4 --seed 4242 --split validation`.

| Policy | Success rate | Collision-risk rate | Height-violation rate | Average return |
| --- | ---: | ---: | ---: | ---: |
| `zero` | `0.0` | `0.015625` | `0.0` | `-88.2416763305664` |
| `random` | `0.0` | `0.00390625` | `0.3671875` | `-195.02862548828125` |
| `oracle` | `1.0` | `0.0` | `0.0` | `2179.718994140625` |
| `trained model_299.pt` | `1.0` | `0.0` | `0.0` | `2106.15966796875` |

Zero and random policies are unsafe/non-controller baselines; their collision-risk or height terminations are expected behavior and confirm reset/accounting paths.

## Regression Notes

During M5 work, M2, M3, and M4 regression audits were rerun and passed. The final reward-accounting patch touched only the independent M5 RL environment and does not modify the Direct or Baseline task entry points.

One Isaac/Kit startup segmentation fault occurred during an earlier trained evaluation attempt before the environment initialized. The exact retry and all subsequent training, evaluation, and audit runs completed successfully.
