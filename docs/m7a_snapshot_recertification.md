# Archived M7A Snapshot Recertification

Archive metadata:

- Source: `/tmp/m7a_snapshot_recertification.md`
- Evidence role: supplementary snapshot recertification evidence
- Archived after implementation commit: `348d0ba1782eedc61c692b5e0558dec04104abab`
- Related independent audit archive: `docs/m7a_independent_audit.md`
- This file does not replace the independent audit report.

---

# M7A Snapshot Recertification

Date: 2026-07-24
Recertifier: M7A independent audit lead (same as original audit)

---

## Original Audit Result

**ACCEPT M7A WITH MAJOR LIMITATION**

Original audit report: `/tmp/m7a_independent_audit.md`

---

## Project State

- Branch: `feature/m7`
- HEAD: `acc27beca2528db21fe1604118e448a87f7e298a` ("Docs: record M6 acceptance with major limitation")
- Base tag: `m6-accepted`

### Git Status

```text
## feature/m7
 M AGENTS.md
 M docs/milestone_state.md
 M scripts/evaluate.py
 M scripts/train.py
 M source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/__init__.py
 M source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py
 M tests/test_env_smoke.py
 M tests/test_m5_ppo_cfg.py
 M tests/test_m6_recurrent_cfg.py
?? docs/m7_experiment_plan.md
?? docs/m7a_verification.md
?? scripts/audit_m7_observation_pipeline.py
?? scripts/audit_m7_pomdp_comparison.py
?? source/uav_rendezvous_rl/uav_rendezvous_rl/observations/
?? source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env.py
?? source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env_cfg.py
?? tests/test_m7_actor_isolation.py
?? tests/test_m7_dropout.py
?? tests/test_m7_fair_ablation.py
?? tests/test_m7_no_future_leakage.py
?? tests/test_m7_noise_seed.py
?? tests/test_m7_observation_delay.py
?? tests/test_m7_observation_partial_reset.py
?? tests/test_m7_sample_hold.py
?? tests/test_m7_task_registration.py
```

`git diff --check`: no output (clean).

Only M7A scope files are modified or untracked. No training artifacts, no M7B/M7C content.

---

## Manifest Comparison

| Manifest | Path | Lines | SHA256 |
|---|---|---|---|
| Auditor (newly generated) | `/tmp/m7a_auditor_manifest.sha256` | 116 | `670391d500acc8d2e5221abb1ffe1082d52e7e1babc635d62144868d5147f60d` |
| Precommit (builder) | `/tmp/m7a_precommit_manifest.sha256` | 116 | `670391d500acc8d2e5221abb1ffe1082d52e7e1babc635d62144868d5147f60d` |
| Source (frozen baseline) | `/tmp/m7a_source_manifest.sha256` | 116 | `670391d500acc8d2e5221abb1ffe1082d52e7e1babc635d62144868d5147f60d` |

**Comparison result: ALL MANIFESTS IDENTICAL.** Zero differences.

---

## Static Critical-File Recheck

| Check | File | Result |
|---|---|---|
| M7A GRU task registered | `tasks/direct/__init__.py` | PASS - 1 occurrence |
| M7A FF task registered | `tasks/direct/__init__.py` | PASS - 1 occurrence |
| Actor uses `p_rel_obs_w` (degraded) | `uav_rendezvous_m7a_env.py` | PASS - 6 occurrences |
| Actor assembly function intact | `uav_rendezvous_m7a_env.py` | PASS - 3 occurrences |
| Push-before-read order intact | `observations/pipeline.py` | PASS - push then read |
| Stage 2 delay config intact | `observations/configs.py` | PASS - delay 1/3 present |
| Stage 1 velocity period intact | `observations/configs.py` | PASS - period 5 present |
| Doc: no history advantage declared | `docs/m7a_verification.md` | PASS |
| Doc: M7B not authorized | `docs/milestone_state.md` | PASS - 2 occurrences |
| No M7B/M7C source code | `source/`, `scripts/`, `tests/` | PASS - zero M7B/M7C files |

All critical assertions from the original audit remain valid.

---

## Audit Integrity

- No project files were modified during this recertification.
- No GPU tasks were run.
- No training, evaluation, or audit scripts were executed.
- M7B and M7C were not entered.
- The current worktree matches the previously audited snapshot exactly.

---

## Conclusion

The worktree at `feature/m7`, HEAD `acc27beca2528db21fe1604118e448a87f7e298a`, is bit-for-bit identical to the snapshot certified in the original M7A independent audit.

The current worktree is the newly certified snapshot for submission.
