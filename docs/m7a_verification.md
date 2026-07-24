# M7A Verification

Date: 2026-07-23

Status: implementation, audits, formal Stage 0/1/2 training, and matched evaluator validation completed. M7A remains in progress pending final review and user acceptance.

## Scope

M7A evaluates controlled partial observability with strictly causal observation degradation only. It does not authorize M7B or M7C work.

## Implemented

- Added M7A tasks `Isaac-Uav-Rendezvous-M7A-GRU-v0` and `Isaac-Uav-Rendezvous-M7A-Feedforward-v0`.
- Added causal observation degradation for delay, velocity sample-and-hold, dropout last-valid hold, and zero-mean Gaussian noise.
- Kept Actor observation at 25D and Critic observation at 57D.
- Added M7A unit/static tests and runtime audits for Stage 3 and Stage 4 infrastructure.
- Added `--m7a_stage` support to training and evaluation scripts.

## Tests And Audits

- Full pytest passed before formal training: `95 passed in 1.65s`.
- M7A Stage 3 10000-step observation pipeline audit passed.
- M7A Stage 4 10000-step observation pipeline audit passed.
- M2, M3, M4, M5, and M6 regression audits passed during M7A implementation.

## Formal Training

All formal runs used `num_envs=256`, `num_steps_per_env=128`, `max_iterations=300`, `seed=42`, `target_motion_mode=Mixed`, and the stated M7A stage.

| Stage | Policy | Checkpoint | Training result |
| --- | --- | --- | --- |
| 0 | GRU | `logs/rsl_rl/uav_rendezvous_m7a_gru/2026-07-23_11-17-57_m7a_stage0_gru_300_seed42/model_299.pt` | final visible success rate `1.0`, zero safety/violation terminations |
| 0 | Feedforward | `logs/rsl_rl/uav_rendezvous_m7a_feedforward/2026-07-23_11-27-45_m7a_stage0_ff_300_seed42/model_299.pt` | final visible success rate `1.0`, zero safety/violation terminations |
| 1 | GRU | `logs/rsl_rl/uav_rendezvous_m7a_gru/2026-07-23_13-40-50_m7a_stage1_gru_300_seed42/model_299.pt` | final visible success rate `1.0`, zero safety/violation terminations |
| 1 | Feedforward | `logs/rsl_rl/uav_rendezvous_m7a_feedforward/2026-07-23_13-50-51_m7a_stage1_ff_300_seed42/model_299.pt` | final visible success rate `1.0`, zero safety/violation terminations |
| 2 | GRU | `logs/rsl_rl/uav_rendezvous_m7a_gru/2026-07-23_14-05-47_m7a_stage2_gru_300_seed42/model_299.pt` | final visible success rate `1.0`, zero safety/violation terminations; some late iterations dipped below `1.0` |
| 2 | Feedforward | `logs/rsl_rl/uav_rendezvous_m7a_feedforward/2026-07-23_14-15-54_m7a_stage2_ff_300_seed42/model_299.pt` | final visible success rate `1.0`, zero safety/violation terminations |

## Matched Validation

Validation used `scripts/evaluate.py` with `num_envs=64`, validation split, seed `4242`, and `target_motion_mode=Mixed`. Stage 1 and Stage 2 used `episodes=8` for `512` episodes per policy. Stage 0 used `episodes=1` for `64` episodes per policy after the comparison script exposed a same-process multi-env lifecycle issue.

| Stage | Policy | Episodes | Success rate | Collision rate | Offset p95 | Relative speed p95 | Return mean | Convergence mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GRU | 64 | `1.0000` | `0.0000` | `0.4630` | `0.2093` | `1953.1494` | `5.6856` |
| 0 | Feedforward | 64 | `1.0000` | `0.0000` | `0.1955` | `0.1050` | `2172.9028` | `3.3537` |
| 1 | GRU | 512 | `1.0000` | `0.0000` | `0.4365` | `0.2088` | `2070.5264` | `4.0795` |
| 1 | Feedforward | 512 | `1.0000` | `0.0000` | `0.3022` | `0.1703` | `2103.0747` | `4.2433` |
| 2 | GRU | 512 | `0.8438` | `0.0000` | `0.4859` | `0.2160` | `1771.1208` | `7.4047` |
| 2 | Feedforward | 512 | `1.0000` | `0.0000` | `0.3054` | `0.1675` | `2151.2178` | `3.5930` |

## Conclusion

No M7A history-value performance advantage is demonstrated. Feedforward outperformed GRU on Stage 0 clean regression, Stage 1 velocity sample-and-hold, and Stage 2 medium delay under the matched validation protocol. Stage 2 especially favored feedforward, with `1.0000` success rate versus GRU `0.8438`, while preserving zero collision risk.

History-value claim status: not allowed.

## Residual Issue

`scripts/audit_m7_pomdp_comparison.py` was diagnosed as unreliable when it attempted to create multiple Isaac Lab Gym environments inside one Isaac app. It was refactored toward shared-env evaluation, but the final trusted formal metrics above come from `scripts/evaluate.py`, which completed reliably for each policy/checkpoint.
