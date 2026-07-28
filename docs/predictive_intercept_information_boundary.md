# Predictive Intercept Information Boundary

## Purpose

This document is the normative PI0 permission matrix for Oracle teacher, causal teacher, student, and future PPO Critic information. A field being available in simulation or stored in a dataset does not make it a valid student input.

## Time Categories

- `current`: information at or before the policy decision timestamp.
- `history`: timestamped causal samples no newer than the decision timestamp.
- `future`: any truth, schedule, command, disturbance, or plan after the decision timestamp.
- `label`: training target stored separately from inference inputs.

No preprocessing, normalization, hidden-state initialization, or feature engineering may indirectly transfer forbidden future information into student inference.

## Permission Matrix

| Information | Oracle | Causal teacher | Student | PI6 Critic |
| --- | --- | --- | --- | --- |
| Current Target truth | allow | allow | forbid; estimates only | allow |
| Historical Target truth | allow | allow | forbid; estimate history only | current-time training context only |
| Complete future Target truth | allow | forbid | forbid | forbid |
| Future maneuver switches | allow | forbid | forbid | forbid |
| Future Target commands | allow | forbid | forbid | forbid |
| Motion mode label | allow | optional current truth for audit | forbid | optional current-time privilege only if explicitly frozen |
| Generator parameters | allow | current/history only if needed | forbid | optional current-time privilege only if explicitly frozen |
| Current Ego truth/measurable state | allow | allow | measurable subset only | allow |
| Current dynamics parameters | allow | allow | forbid unless physically measured and approved | allow |
| Current wind/disturbance truth | allow | allow | forbid unless measured and approved | allow |
| Future wind/disturbance | allow only for explicitly scoped Oracle planning | forbid | forbid | forbid |
| Current observation estimate | optional | optional | allow | allow |
| Estimate confidence/uncertainty | optional | optional | allow | allow |
| Valid mask, age, jump flag | optional | optional | allow | allow |
| Nonzero offset configuration | allow | allow | allow | allow |
| Previous/executed action state | allow | allow | allow when interface is frozen | allow |
| Oracle `T_star` | output/label | forbid as input | forbid as input | forbid as input |
| Oracle planned trajectory | output/label | forbid as input | forbid as input | forbid as input |
| Causal-teacher action/trajectory | N/A | output/label | forbid as inference input | forbid as future input |

## Oracle Boundary

Only Oracle-specific code may invoke the future Target query interface. Oracle use must be explicit in configuration and logs. Future truth may support planning labels and upper-bound evaluation, but Oracle actions must never be mislabeled as deployable.

The future query must not be attached to a generic environment observation dictionary or a shared object accessible to student modules.

## Causal Teacher Boundary

The causal teacher receives current/history truth to isolate prediction and planning quality. It must construct its own causal forecast. It may not reuse cached Oracle trajectories, Oracle-selected intercept times, future mode transitions, future generator state, or future disturbance samples.

Calling the causal teacher and Oracle from the same data-generation process does not permit shared future-bearing feature objects. Their inputs and outputs require separate typed/schema groups.

## Student Boundary

Student inference may contain:

- Estimated `p_rel_w` and `v_rel_w`.
- Causal history of deployable estimates.
- Measurable Ego velocity, attitude representation, and angular rate.
- Nonzero `b_des_w` and offset magnitude.
- Confidence or uncertainty fields.
- Measurement-valid mask.
- Observation age.
- Jump or innovation-rejection flag.
- Previous policy or executed action state after interface freeze.

Student inference must not contain:

```text
target_future_truth
target_future_commands
target_true_trajectory_future
future_segment_schedule
future_mode_switch
target_motion_mode
target_generator_parameters
oracle_intercept_time
oracle_intercept_point
oracle_planned_trajectory
oracle_reference_action
teacher_feasibility_label
future_wind
future_gust
```

Teacher quantities may be supervised labels. Labels must not be concatenated with student inputs, copied into recurrent initial state, used to compute student-only normalization statistics, or loaded by deployment inference code.

## PI6 Actor-Critic Boundary

The Actor is the student and keeps the student boundary unchanged.

The Critic may use current-time training privileges such as:

- Current Target and Ego truth.
- Current estimation error.
- Current validity/confidence state.
- Current dynamics randomization parameters.
- Current wind.
- Current execution delay.

The Critic may not use future Target truth, future motion segments, Oracle future plans, future disturbances, or future commands. Asymmetric Critic privilege does not authorize a future-informed value target input.

## Dataset Separation

Every dataset version must physically or logically separate:

```text
student_inputs/*
oracle_labels/*
causal_teacher_labels/*
truth_current/*
truth_history/*
truth_future/*
metadata/*
```

Student readers must select `student_inputs/*` by allowlist, never by excluding a few known labels from a broad sample dictionary.

Normalization statistics for student inference must be computed only from allowed student input fields and training split records. Future labels and validation/test records are excluded.

## Module Boundary Plan

After implementation authorization:

- `teacher/oracle_teacher.py` may import the future-query interface.
- Oracle-owned intercept search may consume future samples only through an Oracle context.
- `student/*`, `estimation/*`, and deployable policy code must not import future-query modules.
- Generic `imitation/*` readers must expose labels and student inputs through separate APIs.
- Task code must not place future truth in ordinary policy or Critic observation groups.

Static import checks and runtime observation-key checks are both required. String scans alone are not sufficient, but they provide defense in depth.

## Strict Causality Test

Construct two episodes with identical current and historical observations but different future Target trajectories. At the current decision time:

- Student input tensors must be bit-identical.
- Student normalization path must be identical.
- Deterministic student action and hidden-state transition must be identical.
- Causal-teacher inputs must be identical.
- Oracle outputs may differ.

This counterfactual test is required before student training can be accepted.

## Checkpoint and Export Audit

Future student checkpoints must be audited for:

- Network input dimension and named schema version.
- Normalizer keys and dimensions.
- Absence of future-label statistics.
- Hidden-state reset behavior.
- Deployment export dependencies.
- Deterministic inference.

An otherwise successful policy is rejected if information leakage is found.

## PI0 Status

This document defines permissions only. No future-query module, student reader, network, checkpoint, or dataset is created in PI0.
