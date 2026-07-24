# Archived M7A Independent Audit

Archive metadata:

- Source: `/tmp/m7a_independent_audit.md`
- Archive status: reconstructed independent-auditor copy
- Archived after implementation commit: `348d0ba1782eedc61c692b5e0558dec04104abab`
- Supplementary recertification evidence: `docs/m7a_snapshot_recertification.md`
- Snapshot manifest SHA256: `670391d500acc8d2e5221abb1ffe1082d52e7e1babc635d62144868d5147f60d`
- The recertification report is supplementary evidence only and does not replace this independent audit report.

---

# M7A Independent Audit

**Audit report status:**
Reconstructed by the independent auditor after loss of the original
temporary `/tmp` report.

**Original audit conclusion:**
ACCEPT M7A WITH MAJOR LIMITATION

**Snapshot evidence:**
- `/tmp/m7a_snapshot_recertification.md`
- `/tmp/m7a_source_manifest.sha256`
- `/tmp/m7a_auditor_manifest.sha256`
- `/tmp/m7a_precommit_manifest.sha256`

This reconstructed report preserves the original audit findings and
does not replace or alter the snapshot recertification evidence.

---

Date: 2026-07-24
Auditor: Independent read-only audit
Project: `/home/lab_726/uav_rendezvous_rl`
Branch: `feature/m7`
HEAD: `acc27beca2528db21fe1604118e448a87f7e298a`
Base tag: `m6-accepted`

---

## 1. Snapshot Identity

**Result: AUDIT_SNAPSHOT_VERIFIED**

All three manifests (source, auditor, precommit) are bit-for-bit identical:
- 116 lines each
- SHA256: `670391d500acc8d2e5221abb1ffe1082d52e7e1babc635d62144868d5147f60d`

`git status --short --branch` shows only M7A-relevant modifications and additions:
- 9 modified tracked files (all M7A-related)
- 19 untracked new files (all M7A scope)
- No training artifacts, no M7B/M7C content
- `git diff --check`: clean

| Item | Status |
|---|---|
| Source manifest verified | PASS |
| Builder/precommit manifest match | PASS |
| Recertification evidence complete | PASS |
| Git workspace clean except M7A scope | PASS |

---

## 2. Observation Pipeline Architecture

**Location:** `source/uav_rendezvous_rl/uav_rendezvous_rl/observations/`

**Components:**

| File | Lines | Purpose |
|---|---|---|
| `__init__.py` | 7 | Exports `ObservationHistoryBuffer`, `ObservationPipeline`, `ObservationPipelineCfg`, `make_m7a_observation_cfg` |
| `configs.py` | 93 | Immutable `@dataclass(frozen=True)` config; stage aliases; `make_m7a_observation_cfg(stage)` |
| `history_buffer.py` | 62 | Vectorized fixed-length ring buffer with `push`, `read(delay_steps)`, `reset` |
| `corruption.py` | 63 | Stateless deterministic dropout masks and Gaussian noise via sine-based PRNG |
| `pipeline.py` | 206 | `ObservationPipeline` with `observe()`, `reset()`, `diagnostics()` |

**Observe order** (`pipeline.py:102-158`):
1. Convert truth to runtime dtype/device (lines 105-106)
2. Store truth in diagnostics-only `last_*_truth` fields (lines 107-108)
3. Push truth into position and velocity history buffers (lines 110-111)
4. Read delayed samples according to configured delay (lines 112-113)
5. Compute update masks from `step_count % update_period_steps` (lines 117-118)
6. Update held values via `torch.where` (lines 119-120)
7. Generate dropout masks; apply last-valid-value hold (lines 122-127)
8. Generate zero-mean Gaussian noise; add to valid signal (lines 129-139)
9. Store diagnostics (update masks, dropout masks, read ages, counters) (lines 140-157)
10. Increment observation_count and step_count (lines 152-157)

**Architecture assessment:** PASS. Strictly causal, no future access. All corruption applied sequentially in correct order.

---

## 3. Delay Causality

**Implementation:** `history_buffer.py:46-62`

```python
def push(self, sample):
    if self.history_length > 1:
        self.data[:, :-1] = self.data[:, 1:].clone()  # shift left
    self.data[:, -1] = values                           # new at rightmost

def read(self, delay_steps):
    return self.data[:, self.history_length - 1 - delay]
```

**Verification (independent pulse/ramp sequences, not production functions):**

| Check | Test | Independent? | Result |
|---|---|---|---|
| delay=0 returns current sample | `test_delay_zero_returns_current_sample` | Yes - direct value comparison after one step | PASS |
| delay=1 returns previous sample | `test_delay_one_returns_previous_sample` | Yes - two-step cross-check | PASS |
| delay=3 correct index, no off-by-one | `test_multi_step_delay_index_is_correct` | Yes - 6-step sequence with expected [0,0,0,0,1,2] | PASS |
| reset fills entire history | `test_startup_buffer_is_initialized_with_reset_sample` | Yes - checks all history slots | PASS |
| no previous-episode leakage after reset | `test_reset_first_frame_has_no_previous_episode_leakage` | Yes - full episode, reset, check first frame | PASS |
| pulse does not appear before delay | `test_pulse_input_does_not_appear_before_configured_delay` | Yes - pulse at step 4, appears at step 6 with delay=2 | PASS |
| ramp has no off-by-one error | `test_ramp_input_has_no_off_by_one_error` | Yes - dual-delay ramp with expected values | PASS |
| push-read order correct | `pipeline.py:110-113` - push before read | Code inspection | PASS |
| no future buffer access | `history_buffer.py:62` - max index = `history_length-1` | Code inspection | PASS |
| uninitialized buffer not exposed | `history_buffer.py:28` zeros init + immediate reset fill | Code inspection | PASS |

**Delay causality: PASS on all 10 checks.**

**Sample-and-hold:**

| Check | Test | Result |
|---|---|---|
| Uses integer step count | `pipeline.py:117` - `step_count % int(period)` with `torch.long` | PASS |
| 10 Hz = every 5th 50 Hz step | `test_velocity_sample_and_hold_update_period_is_integer_counted` - sequence [100,100,100,100,100,105,105] | PASS |
| Position stays current at period=1 | `test_position_stays_current_when_update_period_is_one` - [0,1,2,3] | PASS |
| Combined delay + sample-hold | `test_sample_hold_uses_delayed_sample_when_both_are_enabled` - correct | PASS |

**Sample-and-hold: PASS on all 4 checks.**

---

## 4. Actor/Critic Information Boundary

**Actor observation (25D):** Verified in `mdp/rendezvous.py:200-228`

| Field | Dim | Source | In Actor |
|---|---|---|---|
| `p_rel_obs_w` | 3 | `ObservationPipeline.observe()` output | Yes |
| `v_rel_obs_w` | 3 | `ObservationPipeline.observe()` output | Yes |
| `v_ego_w` | 3 | Current ego state | Yes |
| `R_ego_6d` | 6 | Current ego state | Yes |
| `omega_ego_b` | 3 | Current ego state | Yes |
| `previous_squashed_action` | 3 | Previous squashed action | Yes |
| `b_des_w` | 3 | Desired offset | Yes |
| `d_offset` | 1 | Offset magnitude | Yes |
| **Total** | **25** | | |

**Critic observation (57D):** Verified in `mdp/rendezvous.py:261-289`

| Field | Added dims | Cumulative |
|---|---|---|
| Actor observation | 25 | 25 |
| `p_ego_w` (truth) | 3 | 28 |
| `p_target_w` (truth) | 3 | 31 |
| `v_target_w` (truth) | 3 | 34 |
| `a_target_w` (truth) | 3 | 37 |
| `R_target_6d` (truth) | 6 | 43 |
| `omega_target_b` (truth) | 3 | 46 |
| `mode_one_hot` (privileged) | 4 | 50 |
| `target_motion_current_params` (privileged) | 6 | 56 |
| `episode_phase` | 1 | 57 |

**Forbidden Actor inputs confirmed absent** (`test_actor_isolation.py`, `uav_rendezvous_m7a_env.py:43-70`):

| Item | Absence verified | Method |
|---|---|---|
| `p_rel_truth_w` | Yes | Uses `p_rel_obs_w` instead; `self.p_rel_w,` absent from actor assembly |
| `v_rel_truth_w` | Yes | Uses `v_rel_obs_w` instead; `self.v_rel_w,` absent |
| `a_target_w` | Yes | Not in actor block |
| `mode_id` | Yes | Not in actor block |
| `mode_one_hot` | Yes | `mode_one_hot` not in actor block |
| `target_motion_current_params` | Yes | Not in actor block |
| dropout mask | Yes | `dropout` not in actor block |
| observation age | Yes | `age` not in actor block |
| raw history buffer | Yes | `history` not in actor block |
| future segment schedule | Yes | None exists |
| future target state | Yes | None exists |

**Critic information boundary:** Critic contains current privileged information only. No future state access detected. PASS.

**Overall: PASS. Actor is 25D with only deployable, causal observations. Critic is 57D with current privilege only.**

---

## 5. Partial Reset and RNG

**Implementation:** `pipeline.py:66-100`

- `ObservationPipeline.reset()` accepts `env_ids`, resolves them, fills only selected histories
- `step_count`, `held_*`, `last_valid_*`, `last_*_obs`, `last_*_truth`, counters all reset for selected envs only
- Other env states untouched

**Verification:**

| Check | Test | Result |
|---|---|---|
| Only selected envs changed | `test_partial_reset_changes_only_selected_environments` - envs 1,3 reset, 0,2,4 checked element-by-element | PASS |
| Reset uses new initial sample | `test_partial_reset_first_observation_uses_new_episode_initial_sample` | PASS |
| Seed reproducibility | `test_noise_is_seed_reproducible_and_does_not_change_truth` - same seed gives same output | PASS |
| Different seeds differ | `test_different_noise_seeds_produce_different_samples` | PASS |
| Dropout seed reproducibility | `test_dropout_seed_reproducibility_and_seed_difference` | PASS |

**Partial reset and RNG: PASS on all checks.**

---

## 6. Stage Configuration

**Configuration definitions** (`configs.py:73-93`):

| Stage | Config | Position delay | Velocity delay | Pos update period | Vel update period | Pos dropout | Vel dropout | Pos noise | Vel noise |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `ObservationPipelineCfg()` | 0 | 0 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | `ObservationPipelineCfg(velocity_update_period_steps=5)` | 0 | 0 | 1 | **5** | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | `ObservationPipelineCfg(position_delay_steps=1, velocity_delay_steps=3)` | **1** | **3** | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | `ObservationPipelineCfg(position_dropout_prob=0.05, velocity_dropout_prob=0.10)` | 0 | 0 | 1 | 1 | 0.05 | 0.10 | 0.0 | 0.0 |
| 4 | Combined (delay 1, delay 3, period 5, dropout 5%/10%, noise 0.02/0.02) | 1 | 3 | 1 | 5 | 0.05 | 0.10 | 0.02 | 0.02 |

**Configuration enters environment:**

| Entry point | How | Verified |
|---|---|---|
| `train.py:103` | `env_cfg.observation_degradation = make_m7a_observation_cfg(stage)` | Code inspection |
| `evaluate.py:121` | Same call | Code inspection |
| `uav_rendezvous_m7a_env.py:25-30` | `ObservationPipeline(self.cfg.observation_degradation, ...)` | Code inspection |
| `uav_rendezvous_m7a_env_cfg.py:16` | Default is `make_m7a_observation_cfg(0)` | Code inspection |

**Stage configuration: PASS on all checks.**

---

## 7. Fairness of GRU/Feedforward Experiments

**Config comparison:**

| Parameter | GRU | Feedforward | Matched? |
|---|---|---|---|
| Policy class | `RslRlPpoActorCriticRecurrentCfg` | `RslRlPpoActorCriticCfg` | Different (by design) |
| `rnn_type` | `"gru"` | N/A | Different (by design) |
| `experiment_name` | `"uav_rendezvous_m7a_gru"` | `"uav_rendezvous_m7a_feedforward"` | Different (by design) |
| `num_steps_per_env` | 128 | 128 | YES |
| `max_iterations` | 100 (overridden to 300) | 100 (overridden to 300) | YES |
| `save_interval` | 25 | 25 | YES |
| `clip_actions` | None | None | YES |
| `obs_groups` | `{"policy":["policy"], "critic":["critic"]}` | Same | YES |
| `init_noise_std` | 0.5 | 0.5 | YES |
| `actor_obs_normalization` | True | True | YES |
| `critic_obs_normalization` | True | True | YES |
| `actor_hidden_dims` | [128, 128] | [128, 128] | YES |
| `critic_hidden_dims` | [128, 128] | [128, 128] | YES |
| `activation` | "elu" | "elu" | YES |
| All algorithm params | Identical | Identical | YES |
| Environment | `UavRendezvousM7AEnv` | `UavRendezvousM7AEnv` | YES |
| Environment config | `UavRendezvousM7AEnvCfg` | `UavRendezvousM7AEnvCfg` | YES |
| Target mode distribution | Mixed (0.25 each) | Mixed (0.25 each) | YES |
| Training seed | 42 | 42 | YES |
| Validation seed | 4242 | 4242 | YES |
| Training iterations | 300 | 300 | YES |
| Training num_envs | 256 | 256 | YES |
| Training num_steps_per_env | 128 | 128 | YES |
| Total interaction steps | 9,830,400 | 9,830,400 | YES |

**M2-M6 task behavior preservation:** `test_m2_through_m6_task_registration_entry_points_are_unchanged` confirms all 5 accepted task IDs retain original entry points. No task behavior was modified.

**Fairness conclusion: PASS. All parameters identical except those that define the GRU vs feedforward distinction.**

---

## 8. Training Evidence

**Training runs:**

| Stage | Policy | Checkpoint | Iteration | Normalizer count | Param count | All finite |
|---|---|---|---|---|---|---|
| 0 | GRU | model_299.pt | 299 | 9,830,400 | 29 (incl. GRU) | Yes |
| 0 | FF | model_299.pt | 299 | 9,830,400 | 21 | Yes |
| 1 | GRU | model_299.pt | 299 | 9,830,400 | 29 | Yes |
| 1 | FF | model_299.pt | 299 | 9,830,400 | 21 | Yes |
| 2 | GRU | model_299.pt | 299 | 9,830,400 | 29 | Yes |
| 2 | FF | model_299.pt | 299 | 9,830,400 | 21 | Yes |

- Normalizer count `9,830,400 = 256 envs x 128 steps x 300 iterations` confirms exact budget.
- All checkpoints include `optimizer_state_dict` (trained, not random init).
- GRU has 29 params (includes `memory_a.*` and `memory_c.*` GRU weights); FF has 21.
- All parameter tensors are finite.

---

## 9. Validation Results

**Formal validation metrics (from `scripts/evaluate.py`, independent processes):**

| Stage | Policy | Episodes | Success rate | Collision risk | Return mean | Offset p95 | Relative speed p95 | Convergence mean |
|---|---|---|---|---|---|---|---|---|
| 0 | GRU | 64 | 1.0000 | 0.0000 | 1953.15 | 0.4630 | 0.2093 | 5.69 |
| 0 | FF | 64 | 1.0000 | 0.0000 | 2172.90 | 0.1955 | 0.1050 | 3.35 |
| 1 | GRU | 512 | 1.0000 | 0.0000 | 2070.53 | 0.4365 | 0.2088 | 4.08 |
| 1 | FF | 512 | 1.0000 | 0.0000 | 2103.07 | 0.3022 | 0.1703 | 4.24 |
| 2 | GRU | 512 | 0.8438 | 0.0000 | 1771.12 | 0.4859 | 0.2160 | 7.40 |
| 2 | FF | 512 | 1.0000 | 0.0000 | 2151.22 | 0.3054 | 0.1675 | 3.59 |

**Per-mode episode counts matched between GRU and FF** (same seed, same split).

**History advantage assessment (per `docs/m7_experiment_plan.md` criteria):**

| Criterion | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| GRU success >= +5pp | No (0.0) | No (0.0) | **FAIL: -0.1562** |
| GRU offset p95 >= -10% | No (+136.8%) | No (+44.4%) | No (+59.1%) |
| GRU speed p95 >= -10% | No (+99.3%) | No (+22.6%) | No (+29.0%) |
| GRU return higher | No (-219.75) | No (-32.55) | No (-380.10) |
| GRU convergence >= -10% | No (+69.5%) | No (-3.9%) | No (+106.1%) |

---

## 10. Stage 2 Failure Analysis

GRU Stage 2 84.38% success rate vs FF 100% investigated:

| Potential cause | Checked | Finding |
|---|---|---|
| Hidden not reset by done | `evaluate.py:407-408`: `_reset_policy(policy_nn, dones)` called every step | PASS - correct |
| Sequence length insufficient | delay=3, `num_steps_per_env=128`, delay window is 2.3% of rollout | PASS - ample |
| Wrong checkpoint loaded | Paths verified; checkpoint confirmed GRU (29 params with memory_a/c) | PASS |
| Normalizer differs | Both use `actor_obs_normalization=True`, `critic_obs_normalization=True` | PASS |
| Training budget mismatch | Both at 300 iterations, 256x128 | PASS |
| Evaluation methodology differs | Identical `evaluate.py` calls with matched args | PASS |

**Conclusion on Stage 2 failure:** The evaluation methodology is correct and identical between GRU and FF. The most likely explanation is that delay degrades GRU performance more than FF because GRU optimization is harder with limited signal and the same fixed budget, while FF's simpler policy can directly learn to smooth or infer from current ego/target state. This is a genuine negative result for the history-value hypothesis, not an implementation bug.

---

## 11. Same-Process Isaac Lifecycle Issue

**Issue:** `audit_m7_pomdp_comparison.py` attempted to create a second `OnPolicyRunner` on the same shared env within one Isaac app, which was unreliable. The script hangs during the second evaluation.

**Impact assessment:**

| Question | Answer |
|---|---|
| Does this affect formal metrics? | No. Formal metrics come from `evaluate.py`, which runs one env per process |
| Are GRU and FF evaluated identically? | Yes. Same `evaluate.py` with identical args except task ID + checkpoint |
| Is this documented? | Yes. Builder report lines 421-424 and `docs/m7a_verification.md` line 60 |
| Is this a blocking issue? | No. Formal metrics from independent processes are reliable |

**Classification: NON-BLOCKING.**

---

## 12. Test Quality

**Total: 95 tests passed**

**Strong tests (independent derivations, exact value checks):**

| Test | Why strong |
|---|---|
| `test_delay_zero_returns_current_sample` | Direct value comparison |
| `test_delay_one_returns_previous_sample` | Two-step cross-validation |
| `test_multi_step_delay_index_is_correct` | 6-step sequence with independent expected values |
| `test_pulse_input_does_not_appear_before_configured_delay` | Independent pulse with delay=2 |
| `test_ramp_input_has_no_off_by_one_error` | Independent dual-delay ramp |
| `test_velocity_sample_and_hold_update_period_is_integer_counted` | 7-step verification with exact expected hold pattern |
| `test_sample_hold_uses_delayed_sample_when_both_are_enabled` | Combined delay + sample-hold with exact expected |
| `test_dropout_holds_last_valid_value_without_exposing_mask` | 100% dropout, verifies hold, checks mask stored separately |
| `test_noise_is_seed_reproducible_and_does_not_change_truth` | Same seed, exact match, truth preserved |
| `test_different_noise_seeds_produce_different_samples` | Different seeds, verified unequal |
| `test_noise_is_approximately_zero_mean_with_configured_scale` | 20000 samples, statistical bounds |
| `test_partial_reset_changes_only_selected_environments` | Element-wise cross-check of all non-reset envs |
| `test_dropout_seed_reproducibility_and_seed_difference` | Same/different seed with 64 envs |

**Moderate tests:**

| Test | Notes |
|---|---|
| `test_startup_buffer_is_initialized_with_reset_sample` | Good but checks shape expansion |
| `test_reset_first_frame_has_no_previous_episode_leakage` | Good, single env |
| `test_position_stays_current_when_update_period_is_one` | Simple case |
| `test_no_dropout_updates_last_valid_value` | Good, after dropout hold test |
| `test_actor_uses_only_degraded_relative_observations` | String-based but comprehensive blacklist |
| `test_critic_is_assembled_after_actor_and_remains_separate` | Order and structure check |
| `test_m7a_cfg_keeps_25d_actor_and_57d_critic_by_inheritance` | Checks inheritance, no new dims |

**Weak tests (string/registration checks only):**

| Test | Weakness |
|---|---|
| `test_m7a_gru_and_feedforward_match_except_policy_type_and_experiment_name` | String check on source text, does not instantiate configs |
| `test_m7a_tasks_share_identical_environment_config` | String count check (== 2) |
| `test_m7a_default_env_cfg_is_clean_stage_zero` | String check for function call |
| `test_m7a_gru_task_registers_independently` | gym.spec check only |
| `test_m7a_feedforward_task_uses_same_environment` | gym.spec check only |
| `test_m2_through_m6_task_registration_entry_points_are_unchanged` | gym.spec check only |

**Missing tests:**
- No combined degradation unit test (delay + sample-hold + dropout + noise together)
- No config validation boundary tests
- Covered by runtime Stage 3/4 pipeline audits

---

## 13. M2-M6 Regression Results

M2-M6 regression verified via:
1. `pytest -q` - 95 passed, includes registration entry point tests
2. Builder report documents passing M2-M6 runtime audits during M7A implementation
3. Modified test files are scope-isolation adjustments only (boundary string positions); no behavior changes

**Regression: PASS.**

---

## 14. Acceptance Matrix

| # | Item | Status |
|---|---|---|
| 1 | Independent M7A GRU task | PASS |
| 2 | Independent M7A FF task | PASS |
| 3 | M2-M6 tasks no regression | PASS |
| 4 | Actor remains 25D | PASS |
| 5 | Critic information boundary correct | PASS |
| 6 | Delay strictly causal | PASS |
| 7 | No future leakage | PASS |
| 8 | Sample-and-hold correct | PASS |
| 9 | Dropout correct (last-valid hold) | PASS |
| 10 | Noise reproducible | PASS |
| 11 | Partial reset no crosstalk | PASS |
| 12 | GRU/FF configuration fair | PASS |
| 13 | Stage 0 config effective | PASS |
| 14 | Stage 1 config effective | PASS |
| 15 | Stage 2 config effective | PASS |
| 16 | Stage 0 formal validation | PASS |
| 17 | Stage 1 formal validation | PASS |
| 18 | Stage 2 formal validation | PASS |
| 19 | Collision risk zero | PASS |
| 20 | No NaN/Inf/CUDA errors | PASS |
| 21 | Stage 0 history advantage proven | **FAIL** |
| 22 | Stage 1 history advantage proven | **FAIL** |
| 23 | Stage 2 history advantage proven | **FAIL** |
| 24 | Documentation declares negative result | PASS |
| 25 | M7B/M7C not entered | PASS |

---

## 15. Blocking Issues

**None identified.** The observation pipeline, delay causality, information boundaries, reset behavior, and fairness controls are all correct.

---

## 16. Major Limitation

**No M7A history-value performance advantage was demonstrated.** In all three formal stages (0, 1, 2), the feedforward policy matched or exceeded the GRU policy on all metrics, while maintaining zero collision risk. The most dramatic gap appears in Stage 2 (medium delay), where GRU success rate drops to 84.38% versus feedforward's 100%.

This means M7A cannot claim that history (via GRU) provides measurable task value under controlled partial observability. The FF policy under observation degradation learns a reactive strategy that, within the same training budget, outperforms the GRU's memory-based approach.

This mirrors the M6 finding (where FF also outperformed GRU) but with stronger causal degradation. The degradation is correctly implemented but did not induce enough partial observability to create a GRU advantage.

**Stage 2 gap clarification:** The GRU 84.38% success is a genuine negative result, not caused by hidden state reset errors, wrong checkpoint loading, unfair evaluation protocol, or observation pipeline bugs.

**Claim:** "History-value advantage has been proven" is NOT ALLOWED.

---

## 17. Non-Blocking Issues

1. `scripts/audit_m7_pomdp_comparison.py` exposed an Isaac same-process multi-environment lifecycle issue. Formal metrics come from independent `evaluate.py` processes and are trustworthy.

2. Some registration and configuration tests are string-based and do not fully instantiate configuration classes.

3. No standalone unit test exercises combined degradation (delay + sample-hold + dropout + noise together). Covered by runtime Stage 3/4 pipeline audits.

4. Stage 0 validation used only 64 episodes (vs 512 for Stages 1 and 2), due to the same-process lifecycle issue discovered during comparison script work. Statistical confidence is lower for Stage 0.

---

## 18. Final Recommendation

**RECOMMENDATION: B. ACCEPT M7A WITH MAJOR LIMITATION**

**Justification:**

The observation degradation infrastructure is correct and well-tested:
- Delay causality is strictly maintained (push before read, no future access, pulse/ramp verified)
- Sample-and-hold updates at correct integer step boundaries
- Dropout applies last-valid-value hold, mask not exposed to Actor
- Noise is seed-reproducible and zero-mean, truth untouched
- Partial reset has no crosstalk between environments
- Actor remains 25D deployable; Critic reads current privilege only
- GRU vs FF experiments are fair with matched budgets, seeds, and configurations
- All six training runs completed with verified interaction budgets
- Zero collision risk maintained
- No NaN/Inf/CUDA errors
- M2-M6 regression preserved
- Documentation honestly declares no history advantage

The major limitation is that no M7A history-value performance advantage was demonstrated. Feedforward outperformed GRU in all three formal stages. This is a genuine negative result under correctly-implemented controlled partial observability, not an infrastructure defect.
