# Predictive Intercept Dataset Schema

## Status

This is the PI0 schema contract for future PI2-PI5 work. PI0 does not generate a dataset, create output directories, or implement readers/writers.

## Design Goals

The dataset must be versioned, replayable, deterministic, coordinate-explicit, split-safe, and resistant to student future leakage. It must support Oracle upper-bound labels, causal-teacher labels, damaged student observations, offline student training, DAgger, and audit replay.

## Dataset Identity

Each dataset root must record:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_name` | string | `uav_predictive_intercept` |
| `schema_version` | semantic version | incompatible changes require major bump |
| `generator_commit` | full Git SHA | implementation identity |
| `config_hash` | SHA-256 | canonical generation configuration |
| `created_utc` | timestamp | provenance only |
| `coordinate_frame` | string | local world-aligned `_w` |
| `policy_dt_s` | float | decision sample period |
| `physics_dt_s` | float | rollout integration period |
| `units_version` | string | unit convention identity |
| `split_manifest_hash` | SHA-256 | fixed train/validation/test assignment |

Minimum units:

```text
position: m
velocity: m/s
acceleration: m/s^2
jerk: m/s^3
angle: rad
angular velocity: rad/s
time: s
```

## Episode Identity

| Field | Type | Shape | Student input |
| --- | --- | --- | --- |
| `episode_id` | string/int64 | scalar | metadata only |
| `scenario_seed` | int64 | scalar | no |
| `reset_index` | int64 | scalar | no |
| `split` | enum | scalar | no |
| `target_motion_family` | enum | scalar | forbidden |
| `target_generator_config_hash` | string | scalar | forbidden |
| `b_des_w` | float32 | `[3]` | yes |
| `b_des_norm_m` | float32 | `[1]` | yes |
| `b_min_m` | float32 | `[1]` | configuration |
| `d_exclusion_m` | float32 | `[1]` | configuration/optional input only if deployed |
| `r_tol_m` | float32 | `[1]` | configuration/optional input only if deployed |

Every episode must validate `||b_des_w|| >= b_min > 0` and `||b_des_w|| > d_exclusion + r_tol + m_geometry`.

## Per-Step Metadata

| Field | Type | Shape |
| --- | --- | --- |
| `step_index` | int64 | scalar |
| `timestamp_s` | float64 | scalar |
| `episode_time_s` | float32 | scalar |
| `terminated` | bool | scalar |
| `truncated` | bool | scalar |
| `termination_reason` | enum | scalar |

Timestamps must be monotonic. Step gaps must either equal the declared sample period or carry an explicit missing-sample record.

## Truth Groups

### Current and Historical Truth

```text
truth_current/p_target_w          float32 [3]
truth_current/v_target_w          float32 [3]
truth_current/a_target_w          float32 [3]
truth_current/p_ego_w             float32 [3]
truth_current/v_ego_w             float32 [3]
truth_current/a_ego_w             float32 [3]
truth_current/j_ego_w             float32 [3]
truth_current/ego_attitude_6d     float32 [6]
truth_current/omega_ego_b         float32 [3]
truth_history/*                   float32 [H_truth, ...]
```

These groups support teacher generation and auditing. They are not automatically student inputs.

### Future Truth

```text
truth_future/timestamps_s         float32 [H_future]
truth_future/p_target_w           float32 [H_future, 3]
truth_future/v_target_w           float32 [H_future, 3]
truth_future/a_target_w           float32 [H_future, 3]
truth_future/mode_switches        versioned variable-length record
truth_future/commands             versioned variable-length record
```

Future truth is Oracle-only label context and is strictly forbidden from student readers, student normalization, PPO Actor inputs, and PPO Critic inputs.

## Student Input Group

The authoritative student reader uses an allowlist under `student_inputs/`:

```text
student_inputs/p_rel_est_w             float32 [H_hist, 3]
student_inputs/v_rel_est_w             float32 [H_hist, 3]
student_inputs/p_rel_sigma             float32 [H_hist, K_p]
student_inputs/v_rel_sigma             float32 [H_hist, K_v]
student_inputs/measurement_valid       bool    [H_hist, 1]
student_inputs/observation_age_s       float32 [H_hist, 1]
student_inputs/jump_flag               bool    [H_hist, 1]
student_inputs/innovation_rejected     bool    [H_hist, 1]
student_inputs/v_ego_w                 float32 [H_hist, 3]
student_inputs/ego_attitude_6d         float32 [H_hist, 6]
student_inputs/omega_ego_b             float32 [H_hist, 3]
student_inputs/previous_action         float32 [H_hist, 3]
student_inputs/b_des_w                  float32 [H_hist, 3]
student_inputs/b_des_norm_m             float32 [H_hist, 1]
student_inputs/history_valid_mask      bool    [H_hist, 1]
```

`K_p` and `K_v` are frozen in PI3 as scalar, diagonal-3, or another explicit uncertainty representation. Fields unavailable from the deployment estimator must not be invented as truth-derived inputs.

The student input history is strictly causal. Padding must use an explicit validity mask and must not contain future samples.

## Estimation Audit Group

```text
estimation/error_position_w            float32 [3]
estimation/error_velocity_w            float32 [3]
estimation/noise_state                  versioned fields
estimation/bias_state                   versioned fields
estimation/delay_steps                  int32 [1]
estimation/dropout_state                bool [1]
estimation/jump_magnitude_m             float32 [1]
estimation/jump_remaining_steps         int32 [1]
estimation/confidence                   float32 [C]
```

This group supports analysis and current-time Critic privilege where authorized. It is not included wholesale in student inputs.

## Oracle Label Group

```text
oracle_labels/feasible                  bool [1]
oracle_labels/infeasible_reason         enum [1]
oracle_labels/intercept_time_s          float32 [1]
oracle_labels/time_to_intercept_s        float32 [1]
oracle_labels/intercept_point_w          float32 [3]
oracle_labels/intercept_point_rel_ego_w  float32 [3]
oracle_labels/reference_timestamps_s     float32 [H_ref]
oracle_labels/reference_position_w       float32 [H_ref, 3]
oracle_labels/reference_velocity_w       float32 [H_ref, 3]
oracle_labels/reference_acceleration_w   float32 [H_ref, 3]
oracle_labels/reference_jerk_w           float32 [H_ref, 3]
oracle_labels/reference_action           float32 [H_ref, 3]
oracle_labels/trajectory_cost            float32 [1]
oracle_labels/constraint_margins         float32 [M]
oracle_labels/rollout_terminal_error_m   float32 [1]
```

These are supervised targets only. `intercept_time`, intercept point, planned trajectory, feasibility, and action labels are forbidden from student inference inputs.

## Causal Teacher Label Group

```text
causal_teacher_labels/feasible
causal_teacher_labels/prediction_model_id
causal_teacher_labels/predicted_target_trajectory
causal_teacher_labels/intercept_time_s
causal_teacher_labels/reference_velocity_w
causal_teacher_labels/reference_action
causal_teacher_labels/constraint_margins
```

Oracle and causal-teacher labels for the same state remain separate. Conflicting labels must be retained with ambiguity metadata; opposite trajectories must not be silently averaged.

## DAgger Fields

```text
dagger/iteration
dagger/rollout_policy_checkpoint_sha256
dagger/query_teacher
dagger/student_action
dagger/causal_teacher_action
dagger/oracle_action_optional
dagger/action_disagreement
dagger/label_ambiguity_flag
dagger/state_visit_weight
```

## Split Rules

- Episode identity, not individual steps, determines split.
- Target maneuver parameter families and seed ranges must not leak across splits where the experiment claims out-of-distribution evaluation.
- Split manifests are immutable and hashed.
- Normalization uses training split student-input fields only.
- Validation/test labels remain available for metrics but never training updates.

## Replay and Determinism

A replay record must contain enough configuration and seed identity to reproduce Target truth, estimation damage, teacher planning, and Ego rollout. Replay acceptance requires:

- Monotonic timestamps.
- Matching schema/config hashes.
- Matching coordinate and units versions.
- Deterministic field equality within declared tolerance.
- Numerical log and visualization agreement.

## Version Compatibility

Readers must reject unsupported major versions with a clear error. Adding optional fields requires a minor version bump; changing meaning, units, frame, shape, or causality requires a major version bump.

## Leakage Audit

Before any student training:

- Enumerate exact student input keys.
- Prove no `truth_future/*` or teacher label key is selected.
- Check normalizer keys and sample counts.
- Perform the counterfactual future-change causality test.
- Inspect serialized batches and checkpoints.

## PI0 Stop

This schema is a design artifact only. Dataset writer/reader code and formal data generation require later explicit authorization.
