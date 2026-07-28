# M7B-S1 Independent Audit Report

Date: 2026-07-26
Auditor: AUDITOR_B (read-only)
Project: `/home/lab_726/uav_rendezvous_rl`
Branch: `feature/m7b`

---

## 1. Audit Snapshot Identity

| Item | Value |
|---|---|
| Branch | `feature/m7b` |
| HEAD | `555daeb598ef633f7e2dbf8b86a12148cd48cf34` |
| HEAD message | "Docs: record M7A acceptance with major limitation" |
| Base tag | `m7a-accepted` |
| Modified tracked files | 13 (all M7B scope) |
| New untracked files | 28 + `docs/m7b_g5_independent_audit.md` |
| `git diff --check` | Clean |
| Initial worktree | Recorded above |
| Final worktree | Identical; only `docs/m7b_s1_independent_audit.md` added by this audit |

---

## 2. Document Control Surface

| Check | Status |
|---|---|
| G5 report conclusion | `ACCEPT M7B-G5 WITH NON-BLOCKING ISSUES` — confirmed in `docs/m7b_g5_independent_audit.md` | PASS |
| G5 non-blocking issues addressed | Missing seed 43 recorded: `m7b_verification.md` lines 39-45; contradictory auth statements resolved: `milestone_state.md` now clean | PASS |
| Current status | `M7B-S1 executed, awaiting independent acceptance` (`milestone_state.md` line 9) | PASS |
| S2/S3/S4 status | "not started" (`m7b_verification.md` line 262) | PASS |
| M7C authorization | "not authorized" (`milestone_state.md` line 7) | PASS |
| `test_env_smoke.py` assertions | Checks `Next executable stage: M7B-S1`, `M7C authorization: not authorized`, `Current executable stage: M7B-S1 dynamics only` | PASS |
| No fragile self-fulfilling assertions | Assertions check against explicit status strings; not circular | PASS |

---

## 3. Preflight and Runtime Audit

### 3.1 Targeted Tests

**Command (per verification doc line 129):**
```bash
pytest tests/test_m7b_*.py tests/test_env_smoke.py -q
```

| Claimed | Independent | Match |
|---|---|---|
| `41 passed in 0.92s` | `41 passed in 0.89s` | YES (timing variance irrelevant) |

### 3.2 Short Startup Audit

Per verification doc lines 136-138: 4 env, 8 step, seed 42, zero policy. Claimed: passed, S1 dynamics active, S2/S3 disabled.

### 3.3 10000-Step Runtime Audit

Per verification doc lines 146-150: 16 env, 10000 step, seed 42, random policy.

| Item | Stated | Verdict |
|---|---|---|
| First attempt failed | `appState='startup'`, `UptimeSeconds='0'` (Isaac infra, before project env init) | PASS — correct attribution |
| Retry same command+seed+args | Confirmed identical retry | PASS |
| Retry result | 25D/65D, finite, 10000 steps, 0 collision/workspace/height | PASS |
| Crash after env init? | No — crash was before environment initialization | PASS (not blocking) |

---

## 4. Training and Checkpoint

### 4.1 Checkpoint Verification

| Item | Value | Verdict |
|---|---|---|
| Path | `logs/rsl_rl/uav_rendezvous_m7b_feedforward/2026-07-26_20-57-39_m7b_s1_ff_300_seed42/model_299.pt` | PASS |
| Size | 563,379 bytes | PASS |
| SHA-256 | `2e159913068a795eef6b48924267b453876f547d6113cffc3c69efaac61de3dc` | PASS |
| `iter` key | 299 | PASS |
| Normalizer count | 9,830,400 = 256 × 128 × 300 | PASS |
| Param keys | 21 (feedforward, not recurrent) | PASS |
| Critic norm shape | `[1, 65]` (65D) | PASS |
| All finite | Yes | PASS |
| Optimizer state present | Yes (trained, not random init) | PASS |

### 4.2 Training Config (from `params/env.yaml` and `params/agent.yaml`)

| Parameter | Value | Verdict |
|---|---|---|
| `seed` | 42 | PASS |
| `num_envs` | 256 | PASS |
| `num_steps_per_env` | 128 | PASS |
| `max_iterations` | 300 | PASS |
| `experiment_name` | `uav_rendezvous_m7b_feedforward` | PASS |
| `run_name` | `m7b_s1_ff_300_seed42` | PASS |
| `tau_velocity_scale` | 1.50 (S1 randomization max) | PASS |
| `acceleration_limit_scale` | 1.20 | PASS |
| `speed_limit_scale` | 1.10 | PASS |
| `linear_drag` | 0.15 | PASS |
| `action_delay_steps` | 0 (S2 disabled) | PASS |
| `steady_wind_max_magnitude` | 0.0 (S3 disabled) | PASS |
| `gust_max_magnitude` | 0.0 (S3 disabled) | PASS |
| Resume flag | Not present | PASS (from-scratch) |

### 4.3 From-Scratch Confirmation

| Check | Status |
|---|---|
| No resume flag in training command (`m7b_verification.md` line 178) | PASS |
| No checkpoint load in training command | PASS |
| Normalizer count exactly 9,830,400 (not cumulative from prior run) | PASS |
| S0 and S1 checkpoints in separate run directories | PASS |
| No GRU training, no extra runs, no checkpoint cherry-picking | PASS |

---

## 5. S1 Stage Isolation

### 5.1 Configuration Verification

| Parameter | S1 status | Source config | Runtime diagnostics | Match |
|---|---|---|---|---|
| tau_velocity_scale | Randomized [0.75, 1.50] | `params/env.yaml`: 1.50 | min=0.765, max=1.441 | PASS |
| acceleration_limit_scale | Randomized [0.80, 1.20] | `params/env.yaml`: 1.20 | min=0.809, max=1.200 | PASS |
| speed_limit_scale | Randomized [0.90, 1.10] | `params/env.yaml`: 1.10 | min=0.900, max=1.100 | PASS |
| linear_drag | Randomized [0.00, 0.15] | `params/env.yaml`: 0.15 | min=0.0002, max=0.149 | PASS |
| action_delay_steps | **Strictly 0** | `params/env.yaml`: 0 | min=0, max=0 | PASS |
| steady_wind | **Strictly 0** | `params/env.yaml`: 0.0 | norm_max=0.0 | PASS |
| gust | **Strictly 0** | `params/env.yaml`: 0.0 | target/current max=0.0 | PASS |
| Observation pipeline | Clean (delay=0, dropout=0, noise=0) | Code | Confirmed in diagnostics | PASS |

### 5.2 Actor Isolation (25D)

| Check | Status |
|---|---|
| Policy observation dim | 25 (confirmed) | PASS |
| No dynamics params in Actor | tau/accel/speed/drag only in Critic and dynamics | PASS |
| No action delay in Actor | delay_steps used for FIFO; not in Actor observation | PASS |
| No wind/gust in Actor | wind used only in dynamics and Critic | PASS |
| No target mode in Actor | mode_id/mode_one_hot only in Critic assembly | PASS |

### 5.3 Critic Isolation (65D)

| Check | Status |
|---|---|
| Critic dim | 65 (confirmed in diagnostics + checkpoint normalizer shape) | PASS |
| Privilege: current only | No future gust, schedules, or target states | PASS |

### 5.4 M2-M7A Preservation

| Check | Status |
|---|---|
| 7 accepted task entry points unchanged | `tasks/direct/__init__.py` unchanged for M2-M7A IDs | PASS |
| d_safe unmodified | Inherited from parent env config | PASS |
| Non-contact objective | No collision reward added; `collision_risk_rate=0` confirmed | PASS |

---

## 6. Independent Validation Reproduction

**Command (exactly as in `m7b_verification.md` lines 209-211):**
```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
  /home/lab_726/IsaacLab/isaaclab.sh -p scripts/evaluate.py \
  --task Isaac-Uav-Rendezvous-M7B-Feedforward-v0 --m7b_stage 1 \
  --policy trained \
  --checkpoint .../m7b_s1_ff_300_seed42/model_299.pt \
  --num_envs 64 --episodes 8 --seed 4242 --split validation \
  --target_motion_mode Mixed --force_mode_cycle_on_reset \
  --determinism_check --device cuda:0 --headless
```

### 6.1 Overall Results

| Metric | Claimed | Independent | Match |
|---|---|---|---|
| Episodes | 512 (128/mode) | 512 (128/mode) | YES |
| Overall success rate | 1.0 | 1.0 | YES |
| Collision risk rate | 0.0 | 0.0 | YES |
| Workspace violation rate | 0.0 | 0.0 | YES |
| Height violation rate | 0.0 | 0.0 | YES |
| Speed violation rate | 0.0 | 0.0 | YES |
| Deterministic max_abs_delta | 0.0 | 0.0 | YES |
| Actor dim | 25 | 25 | YES |
| Critic dim | 65 | 65 | YES |
| Finite diagnostics | true | true | YES |
| **Successful offset p95** | **0.3154950439929962** | **0.3154950439929962** | **BIT-EXACT** |
| **Successful speed p95** | **0.16483217477798462** | **0.16483217477798462** | **BIT-EXACT** |
| **Average return** | **2119.27783203125** | **2119.27783203125** | **BIT-EXACT** |
| **Convergence time p95** | **5.199999809265137** | **5.199999809265137** | **BIT-EXACT** |

### 6.2 Per-Mode Results

| Mode | Episodes | Claimed success | Independent success | Claimed offset p95 | Independent offset p95 | Match |
|---|---|---|---|---|---|---|
| ConstantAcceleration | 128 | 1.0 | 1.0 | 0.31992992758750916 | 0.31992992758750916 | **BIT-EXACT** |
| ConstantTurn | 128 | 1.0 | 1.0 | 0.31152456998825073 | 0.31152456998825073 | **BIT-EXACT** |
| ConstantVelocity | 128 | 1.0 | 1.0 | 0.3157779276371002 | 0.3157779276371002 | **BIT-EXACT** |
| PiecewiseAcceleration | 128 | 1.0 | 1.0 | 0.3155469000339508 | 0.3155469000339508 | **BIT-EXACT** |

### 6.3 Gate Verdicts

| Gate | Threshold | Actual | Verdict |
|---|---|---|---|
| Overall success_rate | ≥ 0.95 | 1.0 | **PASS** |
| Per-mode success_rate | ≥ 0.90 | 1.0 (all 4) | **PASS** |
| Collision risk rate | = 0 | 0.0 | **PASS** |
| Workspace violation rate | = 0 | 0.0 | **PASS** |
| Height violation rate | = 0 | 0.0 | **PASS** |
| Speed violation rate | = 0 | 0.0 | **PASS** |
| Deterministic max_abs_delta | = 0 | 0.0 | **PASS** |
| Finite diagnostics | = true | true | **PASS** |
| Offset p95 | ≤ 0.50 m | 0.3155 | **PASS** |
| Speed p95 | ≤ 0.30 m/s | 0.1648 | **PASS** |
| Actor dim | = 25 | 25 | **PASS** |
| Critic dim | = 65 | 65 | **PASS** |
| S1 dynamics randomized | Yes | tau/accel/speed/drag varies | **PASS** |
| S2 delay disabled | Strictly 0 | min=max=0 | **PASS** |
| S3 wind/gust disabled | Strictly 0 | norm_max=0.0 | **PASS** |

**All 15 gates PASS.**

---

## 7. Blocking and Non-Blocking Issues

### Blocking Issues

**None.**

### Non-Blocking Issues

1. **10000-step runtime audit first attempt crashed at Isaac infra level** (`appState='startup'`, `UptimeSeconds='0'`). The crash occurred before project environment initialization and matches historical pattern from M6 (known_issues.md). Retry with identical command passed. Not attributable to M7B project code. Not blocking.

2. **Worktree diff audit scope**: Between G5 audit and S1 audit, 2 additional files (`docs/implementation_plan.md`, `docs/known_issues.md`) were modified. These are M7B documentation updates and do not affect source code or evaluation behavior.

---

## 8. Final Verdict

**ACCEPT M7B-S1**

**Justification:**

All 15 pre-registered S1 gates pass with bit-exact metric reproduction. The checkpoint is confirmed from-scratch with the locked protocol (256 envs, 300 iterations, seed 42, feedforward only). S1 dynamics randomization is active with correct ranges; S2 delay and S3 wind/gust are strictly disabled. Actor 25D and Critic 65D isolation confirmed. M2-M7A behavior preserved. Zero violations across 512 episodes with 100% success.

No blocking issues. One non-blocking infra-level segfault retry noted.

**M7B-S2 may proceed after user acknowledgment.**

---

## 9. Report Metadata

| Item | Value |
|---|---|
| Report | `docs/m7b_s1_independent_audit.md` |
| Auditor | AUDITOR_B (read-only) |
| Branch | `feature/m7b` |
| HEAD | `555daeb` |
| Checkpoint SHA-256 | `2e159913068a795eef6b48924267b453876f547d6113cffc3c69efaac61de3dc` |
| Tests run | 41 passed (targeted M7B + smoke) |
| S1 eval reproduced | Bit-exact on all metrics (overall + per-mode) |
| Project files modified during audit | None (only this report added) |
| GPU tasks run | S1 validation (one evaluate.py process) |
| M7C entered | No |
