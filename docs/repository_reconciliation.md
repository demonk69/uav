# Repository Reconciliation

## Purpose

This document reconciles the committed Legacy repository with the new Predictive Intercept program. It is a PI0 governance record, not an implementation plan approval for PI1.

## Audit Sources

- Read-only repository audit: `/tmp/pi0_repository_audit.md`.
- Read-only M7B freeze audit: `/tmp/m7b_freeze_audit.md`.
- M7B G5 independent audit: `docs/m7b_g5_independent_audit.md`.
- M7B S1 independent audit: `docs/m7b_s1_independent_audit.md`.
- User baseline and branch decisions recorded before PI0 implementation.

The `/tmp` reports are external audit evidence and are not part of the M7B freeze commit. Their conclusions and the resulting repository identities are recorded here.

## Pre-Freeze State

Before M7B freeze:

| Item | State |
| --- | --- |
| Branch | `feature/m7b` |
| HEAD | `555daeb598ef633f7e2dbf8b86a12148cd48cf34` |
| Historical tag | `m7a-accepted` |
| M7B commits after M7A | none |
| M7B tracked modifications | 13 files |
| M7B new files | 25 files |
| M7B training artifacts | local and Git-ignored |
| Remote M7B branch | absent |

`m7a-accepted` was the last committed and pushed state at audit time, but the user explicitly rejected it as the final Predictive Intercept Legacy freeze baseline. It remains a historical M7A acceptance point.

## M7B Freeze Decision

The user authorized one freeze commit containing exactly the 38 files approved by `/tmp/m7b_freeze_audit.md`. Logs, checkpoints, TensorBoard events, PI0 documents, and PI implementation files were excluded.

Freeze result:

```text
branch: feature/m7b
tag: m7b-accepted
commit: fc2cb17315e7f6b2b8dee463db5ab73f931c938e
remote: origin
```

Verification performed before PI branch creation:

| Identity | Resolved commit |
| --- | --- |
| local `feature/m7b` HEAD | `fc2cb17315e7f6b2b8dee463db5ab73f931c938e` |
| local `m7b-accepted^{commit}` | `fc2cb17315e7f6b2b8dee463db5ab73f931c938e` |
| `origin/feature/m7b` | `fc2cb17315e7f6b2b8dee463db5ab73f931c938e` |
| remote `m7b-accepted^{}` | `fc2cb17315e7f6b2b8dee463db5ab73f931c938e` |

The annotated tag object has its own object ID; the authoritative baseline is the commit obtained by dereferencing the tag with `^{commit}`.

## PI Branch

The Predictive Intercept branch is:

```text
feature/pi
```

It was created only after the M7B worktree was clean and directly from:

```text
m7b-accepted^{commit}
```

The branch was pushed to `origin`. PI0 changes remain uncommitted pending independent PI0 review and user acceptance.

## Frozen Legacy Tasks

The following task IDs and their meanings are frozen at `m7b-accepted`:

```text
Isaac-Uav-Rendezvous-Direct-v0
Isaac-Uav-Rendezvous-Baseline-v0
Isaac-Uav-Rendezvous-RL-v0
Isaac-Uav-Rendezvous-Recurrent-v0
Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0
Isaac-Uav-Rendezvous-M7A-GRU-v0
Isaac-Uav-Rendezvous-M7A-Feedforward-v0
Isaac-Uav-Rendezvous-M7B-Feedforward-v0
Isaac-Uav-Rendezvous-M7B-GRU-v0
```

Predictive Intercept work must not modify their rewards, success criteria, terminations, Actor/Critic observations, action spaces, task IDs, checkpoint interpretation, or acceptance metrics.

M7B-S0 clean and M7B-S1 dynamics-only were completed and independently accepted. M7B-S2, M7B-S3, M7B-S4, and M7C were not executed. The freeze does not claim results for those unexecuted stages.

## Package Separation

Legacy package:

```text
source/uav_rendezvous_rl/
```

Planned Predictive Intercept package:

```text
source/uav_predictive_intercept/
└── uav_predictive_intercept/
```

The new package name is frozen by PI0, but PI0 does not create it. Package creation requires explicit PI1 authorization.

## Artifact Treatment

M7B training logs and checkpoints remain local and Git-ignored. The accepted S0 and S1 checkpoints are identified by path and SHA-256 in `docs/m7b_verification.md`. They are not repository baseline objects and do not replace the freeze commit identity.

Predictive Intercept outputs will use the separate planned namespace:

```text
outputs/predictive_intercept/
```

PI0 does not create output directories or datasets.

## Reconciliation Conclusion

- Public and local Legacy identities are reconciled.
- M7B is committed, pushed, tagged, and cleanly separated from PI0.
- `m7a-accepted` is historical; `m7b-accepted` is authoritative for Predictive Intercept.
- `feature/pi` has the exact required base.
- No Predictive Intercept code or package exists during PI0.
- PI1 remains unauthorized.
