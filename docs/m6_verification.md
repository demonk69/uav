# M6 Verification

Date: 2026-07-23

Status: implementation complete and locally verified, pending user acceptance.

## Scope

M6 adds recurrent PPO and implicit target-motion history use on top of the accepted M5 simplified RL task. It does not enter M7 and does not add wind, dynamics randomization, observation noise, delays, dropped observations, high-fidelity UAV assets, Pegasus, PX4, ROS 2, cameras, or first-stage perception.

Verified constraints:

- Recurrent task is independent: `Isaac-Uav-Rendezvous-Recurrent-v0`.
- M5 feedforward task `Isaac-Uav-Rendezvous-RL-v0` remains feedforward and is not converted to recurrent PPO.
- Fair M6 feedforward ablation task is independent: `Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0`.
- Actor observation remains 25D deployable current-state information.
- Critic observation remains 57D privileged current-state information under the separate `critic` group.
- Actor is not given target motion mode, generator parameters, complete future trajectories, future target states, or future target commands.
- Recurrent policy uses `ActorCriticRecurrent` with GRU actor and critic memories.
- Inference calls `policy.reset(dones)` after environment steps so recurrent hidden state is reset per completed environment.

## Commands

All Isaac Lab commands were run through `/home/lab_726/IsaacLab/isaaclab.sh` with `PYTHONPATH`, `PYTHONHOME`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, and `VIRTUAL_ENV` cleared.

## Training Runs

Each formal M6 training run used 256 environments, 128 rollout steps per environment, 300 PPO iterations, seed `42`, and `cuda:0`.

| Run | Task | Target mode | Timesteps | Checkpoint |
| --- | --- | --- | ---: | --- |
| CV-only GRU | `Isaac-Uav-Rendezvous-Recurrent-v0` | `ConstantVelocity` | `9,830,400` | `/home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m6_gru/2026-07-22_23-46-03_m6_cv_gru_300_seed42/model_299.pt` |
| Mixed GRU | `Isaac-Uav-Rendezvous-Recurrent-v0` | `Mixed` | `9,830,400` | `/home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m6_gru/2026-07-22_23-56-05_m6_mixed_gru_300_seed42/model_299.pt` |
| Mixed feedforward ablation | `Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0` | `Mixed` | `9,830,400` | `/home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m6_feedforward_ablation/2026-07-23_00-11-43_m6_ff_ablation_300_seed42/model_299.pt` |

Final visible training metrics for the mixed GRU and feedforward ablation both reported `Metrics/success_rate=1.0000` with zero collision, workspace, height, speed, attitude, non-finite, and target-motion-invalid terminations.

## Formal Validation Protocol

CV-only validation used validation split, seed `4242`, deterministic actor, 64 environments, and 4 episodes per environment for 256 ConstantVelocity episodes.

Mixed GRU and feedforward-ablation validation used validation split, seed `4242`, deterministic actor, `--num_envs 64 --episodes 8 --force_mode_cycle_on_reset --determinism_check`, for 512 total episodes and exactly 128 episodes per mode.

All formal validation runs reported finite tensors, 25D policy observations, 57D critic observations, and deterministic repeat `max_abs_delta=0.0`.

## CV-Only GRU Gate

| Metric | Value | Gate |
| --- | ---: | --- |
| Success rate | `0.9921875` | pass, `>= 0.90` |
| Collision-risk rate | `0.0` | pass, `= 0` |
| Successful offset error p95 | about `0.452 m` | pass, `< 0.50 m` |
| Successful relative speed p95 | about `0.212 m/s` | pass, `< 0.30 m/s` |

## Mixed GRU Validation

Overall 512-episode result:

| Metric | Value |
| --- | ---: |
| Success count | `511 / 512` |
| Success rate | `0.998046875` |
| Collision-risk rate | `0.0` |
| Workspace-violation rate | `0.0` |
| Height-violation rate | `0.0` |
| Speed-violation rate | `0.0` |
| Average return | `1985.9753` |
| Successful offset error p95 | `0.4626103044 m` |
| Successful relative speed p95 | `0.2197498679 m/s` |
| Final offset error p95 | `0.4499618113 m` |
| Final relative speed p95 | `0.1608095318 m/s` |
| Convergence time mean | `5.3128 s` |
| Convergence time p95 | `9.7700 s` |

Per-mode result:

| Mode | Episodes | Success rate | Successful offset p95 | Successful relative speed p95 |
| --- | ---: | ---: | ---: | ---: |
| ConstantVelocity | `128` | `1.0` | `0.4665753245 m` | `0.2219980508 m/s` |
| ConstantAcceleration | `128` | `1.0` | `0.4564321339 m` | `0.2197680622 m/s` |
| ConstantTurn | `128` | `1.0` | `0.4640138447 m` | `0.2060891390 m/s` |
| PiecewiseAcceleration | `128` | `0.9921875` | `0.4529049695 m` | `0.2228991538 m/s` |

## Fair Feedforward Ablation Validation

Overall 512-episode result:

| Metric | Value |
| --- | ---: |
| Success count | `512 / 512` |
| Success rate | `1.0` |
| Collision-risk rate | `0.0` |
| Workspace-violation rate | `0.0` |
| Height-violation rate | `0.0` |
| Speed-violation rate | `0.0` |
| Average return | `2192.3433` |
| Successful offset error p95 | `0.1983923167 m` |
| Successful relative speed p95 | `0.1089752316 m/s` |
| Final offset error p95 | `0.0695706010 m` |
| Final relative speed p95 | `0.0033692955 m/s` |
| Convergence time mean | `3.2294 s` |
| Convergence time p95 | `4.1800 s` |

Per-mode result:

| Mode | Episodes | Success rate | Successful offset p95 | Successful relative speed p95 |
| --- | ---: | ---: | ---: | ---: |
| ConstantVelocity | `128` | `1.0` | `0.1931395382 m` | `0.1095632240 m/s` |
| ConstantAcceleration | `128` | `1.0` | `0.1942542940 m` | `0.1089869067 m/s` |
| ConstantTurn | `128` | `1.0` | `0.1961986721 m` | `0.1056272388 m/s` |
| PiecewiseAcceleration | `128` | `1.0` | `0.2089985907 m` | `0.1108132899 m/s` |

## GRU vs Feedforward Comparison

| Metric | Mixed GRU | Fair feedforward ablation | Better |
| --- | ---: | ---: | --- |
| Success rate | `0.998046875` | `1.0` | Feedforward |
| Successful offset p95 | `0.4626103044 m` | `0.1983923167 m` | Feedforward |
| Successful relative speed p95 | `0.2197498679 m/s` | `0.1089752316 m/s` | Feedforward |
| Final offset p95 | `0.4499618113 m` | `0.0695706010 m` | Feedforward |
| Final relative speed p95 | `0.1608095318 m/s` | `0.0033692955 m/s` | Feedforward |
| Average return | `1985.9753` | `2192.3433` | Feedforward |
| Convergence time mean | `5.3128 s` | `3.2294 s` | Feedforward |

M6 recurrent training and hidden-state management are functional, but a measurable implicit-prediction advantage over the fair feedforward baseline was not demonstrated.

## History-Sensitivity Audit

Command used the final mixed GRU checkpoint, the final feedforward-ablation checkpoint, 16 synthetic environments, 16 history steps, and seed `4242`.

The audit verifies that paired synthetic histories end in identical final Actor observations while producing different GRU hidden states and actions. Feedforward actions are identical for identical final observations.

| Pair | Final obs max diff | Actor hidden distance | Critic hidden distance | GRU action distance | Feedforward action distance | GRU oracle-action error | Feedforward oracle-action error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accelerating vs decelerating | `0.0` | `24.7778644562` | `8.2447977066` | `2.9858984947` | `0.0` | `1.6854380965` | `0.0585985929` |
| Positive turn vs negative turn | `0.0` | `19.5306282043` | `7.6755094528` | `5.6774120331` | `0.0` | `3.5478774309` | `0.0585985929` |
| PWA previous segment difference | `0.0` | `14.2613744736` | `2.0547432899` | `0.5186043978` | `0.0` | `2.2155265808` | `0.0585985929` |

Result: recurrent hidden state and action are history-sensitive under identical final observations, so the GRU path can use observation history. This audit does not show a performance advantage, and the synthetic current-state oracle comparison favors the feedforward policy.

## Checkpoint Resume Audit

Formal resume audit used the mixed GRU checkpoint, 64 environments, seed `42`, and 3 resumed PPO iterations.

| Metric | Value |
| --- | ---: |
| Save iteration | `299` |
| Loaded iteration | `299` |
| Resumed final iteration | `301` |
| Resume iterations | `3` |
| Optimizer state exists | `true` |
| Actor hidden initial norm | `0.0` |
| Critic hidden initial norm | `0.0` |
| Parameter change norm | `1.1829556227` |
| Reward steps | `24576` |
| Done count | `25` |

The resumed iterations produced finite loss dictionaries for all 3 updates. This verifies checkpoint load, optimizer-state restoration, empty hidden state before rollout, finite resumed training, and parameter updates after resume.

## Runtime And Regression Verification

Static checks:

| Check | Result |
| --- | --- |
| `git diff --check` | pass |
| `py_compile` for changed M6 scripts/source | pass |
| Full pytest suite | pass, `68 passed` |

Milestone runtime regressions:

| Area | Command summary | Result |
| --- | --- | --- |
| M2 Direct runtime | `audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42` | pass, `passed=true`, finite, no contact by center distance |
| M3 target motion | `audit_m3_motion_runtime.py --num_envs 16 --steps 5000 --seed 42 --split train` | pass, finite, four mode counts equal, no Actor leakage |
| M4 baseline | `audit_m4_baseline_runtime.py --num_envs 64 --episodes 5 --seed 42 --split train` | pass, all acceptance checks true |
| M5 RL runtime | `audit_m5_rl_runtime.py --scenario all --num_envs 64 --steps 10000 --seed 42 --split train` | pass, oracle success `1.0`; zero/random finite with expected unsafe-controller terminations |
| M5 trained validation | accepted M5 checkpoint, validation split, seed `4242`, 64 envs, 4 episodes | pass, success `1.0`, collision risk `0`, success offset p95 `0.3592815101 m`, success relative speed p95 `0.1695371419 m/s` |
| M6 recurrent runtime | `audit_m6_recurrent_runtime.py --num_envs 8 --steps 64 --seed 42` | pass, `ActorCriticRecurrent`, actor/critic GRU, actor input 25, critic input 57, partial hidden reset done count 1 |
| M6 recurrent play | `play.py --audit_hidden_state --num_envs 4 --steps 2 --seed 4242` with mixed GRU checkpoint | pass, hidden-state reset audit passed and deterministic play completed |

## Tooling Fixes During Final Verification

Two audit-script issues were found and fixed during final verification:

- `audit_m6_history_sensitivity.py` originally loaded the recurrent and feedforward checkpoints by creating two live Isaac environments in one process. The script now reuses the recurrent audit environment when constructing the feedforward runner.
- `audit_m6_checkpoint_resume.py` originally resumed with `log_dir=None`; RSL-RL's learn path expects a log directory for code-state bookkeeping. The script now uses a temporary directory under `/tmp/opencode` for the resumed runner.

One Isaac Sim startup segmentation fault occurred during a diagnostic history-sensitivity rerun before environment initialization. No external Isaac Lab, Isaac Sim, Pegasus, system dependency, driver, or project dependency files were modified. Subsequent reruns and all formal validation/regression commands completed successfully.

## Final M6 Status

M6 implementation is complete pending user acceptance.

Absolute recurrent PPO functionality, safety, validation, hidden-state reset, checkpoint resume, play/evaluate, and M2-M5 regressions passed. The mixed-mode GRU did not outperform the fair feedforward ablation, so no implicit-prediction performance advantage is claimed.

M7 remains unauthorized.
