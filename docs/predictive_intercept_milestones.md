# Predictive Intercept Milestones

## 0. Authority and Stage Control

This is the authoritative PI0 roadmap for the UAV Predictive Intercept program. It defines PI0 through PI7 so interfaces and audits can be planned end to end.

Describing a later milestone does not authorize it.

The only currently authorized milestone is:

```text
PI0: repository alignment, Legacy freeze, governance, and design definition
```

PI1 through PI7 are interface definitions and future audit criteria only. No later implementation, formal training, dataset generation, commit, merge, tag, or push is authorized by this document.

Every milestone requires, in order:

1. Read-only pre-audit.
2. User confirmation of exact implementation scope.
3. Implementation on the authorized branch.
4. Independent read-only review.
5. User acceptance.
6. Separate user decision for commit, merge, and entry to the next stage.

Implementation self-tests are not independent acceptance.

## 1. Legacy and Predictive Intercept Separation

### 1.1 Legacy Program

The final Legacy baseline is:

```text
tag: m7b-accepted
commit: fc2cb17315e7f6b2b8dee463db5ab73f931c938e
```

`m7a-accepted` at `555daeb598ef633f7e2dbf8b86a12148cd48cf34` remains a historical M7A acceptance point.

The frozen Legacy tasks are:

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

Their rewards, success criteria, terminations, observations, actions, IDs, checkpoint meanings, and accepted metrics must not change during Predictive Intercept work.

### 1.2 New Program

Predictive Intercept uses independent milestones:

```text
PI0 -> PI1 -> PI2 -> PI3 -> PI4 -> PI5 -> PI6 -> PI7
```

It must not be called M7B or M7C. Its planned independent package is:

```text
source/uav_predictive_intercept/
└── uav_predictive_intercept/
```

PI0 records the package plan but does not create it.

## 2. Safety and Research Boundary

The task is non-contact virtual-offset crossing.

The offset must satisfy:

```text
||b_des_w|| >= b_min > 0
```

Zero offset is forbidden. Target entity contact, impact, or center crossing is never success. A hard Target exclusion zone remains independent from the virtual goal.

Geometry must satisfy:

```text
||b_des_w|| > d_exclusion + r_tol + m_geometry
```

If virtual crossing and Target exclusion occur in the same simulation step, exclusion failure takes precedence.

The program must not become contact terminal guidance, physical interception, or collision optimization.

## 3. Scenario Definition

Ego begins near ground level with near-zero velocity. Target is already airborne and generally horizontally separated:

```text
z_ego_0 ~= z_ground + h_init
z_target_0 > z_ego_0
d_xy_0 > 0
```

The representative task is an oblique climbing intercept with horizontal lead. Future training and validation must randomize Target altitude, horizontal range/bearing, Target velocity, motion mode and parameters, Ego heading, and nonzero offset configuration.

The scenario must not collapse to Ego directly beneath a horizontal constant-velocity Target with only vertical motion required.

Before high-fidelity dynamics, "ground takeoff" means near-ground initialization and a 3D velocity command through simplified dynamics. It does not claim motor, ground-effect, landing-gear, attitude-loop, or true liftoff fidelity.

## 4. Spatiotemporal Intercept Definition

World-aligned definitions:

```text
p_rel_w(t) = p_target_w(t) - p_ego_w(t)
v_rel_w(t) = v_target_w(t) - v_ego_w(t)
g_w(t) = p_target_w(t) + b_des_w
```

An Oracle selects `T_star > t_0`. The ideal event is:

```text
p_ego_w(T_star) = g_w(T_star)
```

Numerical crossing uses `r_tol`. Success does not require Target velocity matching, relative velocity approaching zero, hovering, holding, or subsequent following.

### 4.1 Continuous Crossing

For one step:

```text
r_0 = p_ego_w,k - g_w,k
r_1 = p_ego_w,k+1 - g_w,k+1
r(lambda) = r_0 + lambda * (r_1 - r_0), lambda in [0, 1]
d_min_step = min ||r(lambda)||
```

A crossing occurs when `d_min_step <= r_tol`. Endpoint-only detection is insufficient.

### 4.2 Entity Exclusion

```text
d_entity = ||p_ego_w - p_target_w||
```

`d_entity < d_exclusion` produces `target_exclusion_violation`. Exclusion takes precedence over crossing success in the same step.

## 5. Decision-Maker Roles

### 5.1 Oracle Teacher

Oracle may read complete future Target truth, future segment switches, current Ego truth, nonzero offset configuration, dynamics constraints, and explicitly scoped current/future simulation disturbances. It selects intercept time and a feasible timed Ego trajectory.

Oracle is used for feasibility upper bound, supervised labels, planning quality, and control isolation. It is never a deployment input.

### 5.2 Causal Privileged Teacher

The causal teacher may read current/history Target truth, current Ego truth, and current dynamics/disturbance state. It may not read future Target truth, future switches, future commands, complete future trajectories, or future disturbances. It must predict causally and plan against its prediction.

### 5.3 Student

The student reads only deployable causal estimates, history, measurable Ego state, nonzero offset, confidence/validity/age/jump metadata, and authorized previous/executed action state.

The student must not read Target future truth, mode labels, generator parameters, future segments, future commands, Oracle intercept time, Oracle trajectory, or simulator-only future information.

Teacher future information may be a label but not an inference field or normalization source.

## 6. Oracle Planning Baseline

The first Oracle is non-neural:

```text
candidate time search
    + quintic trajectory
    + reference velocity-command mapping
    + actual dynamics rollout
    + feasibility checks
    + cost selection
```

For `T_i = t_0 + tau_i`, define candidate goal:

```text
g_i = p_target_w(T_i) + b_des_w
```

Generate a quintic polynomial per axis from current position, velocity, and acceleration to terminal position `g_i`. Terminal velocity/acceleration are free with regularization or use loose references; Target velocity matching is not enforced.

Derive reference velocity, acceleration, and jerk. For the planned initial first-order velocity dynamics:

```text
v_cmd_ref ~= v_ego_w + tau_v * a_ref_w
```

Apply command and physical limits, then re-roll through the actual environment dynamics. A polynomial is not declared feasible from analytic values alone.

Check speed, acceleration, jerk, workspace, altitude, entity exclusion, numerical stability, action interface, and terminal error.

The first cost includes weighted intercept time, terminal error, terminal speed regularization, integrated acceleration, integrated jerk, and command-rate cost. Select the minimum-cost feasible candidate and report explicit infeasible reasons when none exists.

Complex trajectory optimizers are comparisons only after the analytic baseline is accepted.

## 7. Student Observation and Network Plan

The PI3-frozen per-frame semantics must include estimated relative position/velocity, uncertainty, valid state, observation age, jump flag, measurable Ego state, previous action, and nonzero offset configuration. The student receives a strictly causal history of length `H_hist`.

The first primary student is a single-layer GRU with initial hidden size 128. A stacked-history MLP is the fair control.

```text
frame encoder
    -> GRU / stacked-history MLP
    -> action head: 3D raw action
    -> trajectory head: H x 3 reference velocity
    -> auxiliary heads: time, goal vector, feasibility, optional uncertainty
```

The environment maps:

```text
v_cmd_w = v_max * tanh(a_raw)
```

The trajectory head is receding-horizon: execute the first command and recompute next cycle. Auxiliary labels structure training and are not fed back as future-truth action inputs.

Distance jumps require estimator gating, filtering, confidence, temporal history, and physical action/acceleration/jerk limits. PPO is not the only protection.

## 8. Reward and Termination Direction

Future PPO fine-tuning may begin with teacher tracking, intercept terminal reward, progress, time cost, action magnitude/smoothness, jerk, and explicit failure penalties. Teacher tracking weight must decay toward zero for autonomous closed-loop training.

The final autonomous objective must not restore Legacy relative-velocity matching, hold rewards, long-term offset following, or future-truth distance rewards.

Failure reasons must include:

```text
timeout
target_exclusion_violation
workspace_violation
height_violation
speed_violation
acceleration_violation
jerk_violation
nan_or_inf
teacher_infeasible
estimation_invalid
missed_intercept
```

## 9. Repository Plan

After authorization, the independent package may contain:

```text
uav_predictive_intercept/
    tasks/direct/
    teacher/
        oracle_teacher.py
        causal_teacher.py
        intercept_time_search.py
        polynomial_trajectory.py
        trajectory_rollout.py
        feasibility.py
        trajectory_cost.py
    student/
        observation_encoder.py
        recurrent_policy.py
        stacked_history_policy.py
        trajectory_head.py
        auxiliary_heads.py
    estimation/
        observation_model.py
        correlated_noise.py
        distance_jump.py
        delay_buffer.py
        confidence_model.py
        innovation_gate.py
    imitation/
        dataset_schema.py
        dataset_writer.py
        dataset_reader.py
        behavior_cloning.py
        dagger.py
    rewards/
    terminations/
    metrics/
    visualization/
    common/
```

Planned scripts use `scripts/predictive_intercept/`; planned outputs use `outputs/predictive_intercept/`. PI0 creates none of these.

# PI0: Governance and Design Freeze

## Goal

Reconcile the repository, freeze Legacy, define the independent task, close PI1-PI7 interfaces and audit methods, define information permissions, and define the future dataset schema. No code or training.

## Authorized Existing Files

```text
AGENTS.md
README.md
docs/milestone_state.md
```

## Authorized New Files

```text
docs/repository_reconciliation.md
docs/predictive_intercept_task_definition.md
docs/teacher_student_architecture.md
docs/predictive_intercept_information_boundary.md
docs/predictive_intercept_dataset_schema.md
docs/predictive_intercept_milestones.md
```

## Required Work

- Record local/remote reconciliation and final Legacy tag/SHA.
- Freeze all Legacy task IDs and semantics.
- Record `feature/pi` and planned package name.
- Freeze nonzero offset, geometry separation, continuous crossing, and exclusion precedence.
- Freeze Oracle, causal teacher, student, and Critic information boundaries.
- Freeze dataset field groups, versioning, replay, splits, and leakage controls.
- Define PI1-PI7 implementation, tests, and acceptance without implementing them.

## Prohibited

- Any `source/` modification.
- Creating `source/uav_predictive_intercept/`.
- Teacher, student, environment, reward, termination, estimator, reader/writer, or training code.
- Empty future interface stubs.
- Training or formal data generation.
- PI1 entry.

## Acceptance

- Public/local baseline and final M7B freeze are unambiguous.
- Legacy and PI tasks cannot be confused.
- PI0-PI7 interfaces are closed enough for later pre-audits.
- Information permissions are explicit.
- Nonzero offset and entity exclusion contracts are explicit.
- Dataset fields support later work without student leakage.
- Changed-file scope exactly matches PI0 authorization.
- Independent review concludes `ACCEPT` or a user-accepted non-blocking result.

After implementation self-tests, stop for independent PI0 review. PI1 remains unauthorized.

# PI1: Oracle Teacher Feasibility Baseline

## Goal

Determine whether the authorized simplified Ego dynamics can plan and execute reasonable spatiotemporal virtual-offset crossings when complete future Target truth is available.

## Planned Implementation

- Create independent package and Oracle environment.
- Create read-only future Target truth query owned by Oracle.
- Implement candidate intercept-time grid.
- Implement quintic trajectory generation.
- Map reference acceleration to bounded velocity commands.
- Re-roll candidates through actual environment dynamics.
- Check constraints, costs, and explicit infeasible reasons.
- Add Oracle evaluation and core unit tests.

## Tests

- Future truth callable only from Oracle paths.
- Student imports cannot transitively import future-query functions.
- Stationary and constant-velocity analytic cases.
- Quintic boundary conditions and derivative correctness.
- Stable `infeasible` behavior.
- Fixed-seed reproducibility.
- Continuous crossing and exclusion precedence.
- Full Legacy regression.

## Acceptance

For each Target mode, report feasible rate, terminal spatial error p50/p95, timing error p50/p95, speed/acceleration/jerk/exclusion violation rates, solve time p50/p95, and infeasible-reason distribution.

PI1 trains no neural network.

# PI2: Teacher Dataset and Visualization

## Goal

Convert the accepted Oracle into a replayable, auditable data-production and visualization system.

## Planned Implementation

- Versioned writer and reader.
- Episode/seed/time identity.
- Target truth history and future groups.
- Ego truth, offset, Oracle feasibility/time/point/trajectory/action/cost/margins.
- Student damaged observation history with confidence, masks, age, and jump fields.
- 3D and time-series visualization of Target, virtual offset, planned Ego, executed Ego, intercept event, commands, and constraints.

## Acceptance

- Write/read round trip.
- Monotonic timestamps.
- Complete replay from identity/config.
- Same-seed consistency.
- Student fields contain no future information.
- Visual and numerical logs agree.
- Incompatible dataset versions fail clearly.

# PI3: Causal Observation and Estimation Error

## Goal

Create strictly causal student inputs aligned with an upstream vision/state estimator, including realistic distance jumps and protection metadata.

## Planned Error Models

- Correlated small noise.
- Slow bias drift.
- Single-frame spikes and multi-frame bias.
- Position-jump-induced velocity spikes.
- Delay, dropout, last-valid hold, and low-rate velocity updates.
- Distance-dependent uncertainty and confidence.
- Validity and anomaly flags.

## Planned Protection

- Innovation gate.
- Finite-state correction.
- Confidence reduction.
- Velocity filtering.
- Observation age, valid mask, and jump flag.

## Acceptance

- Every error is strictly causal and independently configurable.
- Target truth is unchanged.
- Random streams are reproducible.
- Requested jump magnitude/duration is accurate.
- Position-to-velocity spike propagation is measured.
- Student input contains no future field.
- Frozen M7A pipeline is not modified.

PI3 does not train the final student.

# PI4: Supervised Student Learning

## Goal

Train the first temporal student from versioned teacher data.

## Planned Models and Losses

- Single-layer GRU, initial hidden size 128.
- Fair stacked-history MLP.
- Current action head.
- Short-horizon reference-velocity head.
- Time-to-intercept, relative goal, and feasibility auxiliary heads.
- Separately logged action, trajectory, time, goal, and feasibility losses.

## Training Rules

- Fixed train/validation/test manifests.
- No Target parameter leakage across claimed splits.
- Student damaged observations only.
- Teacher quantities are labels only.
- Checkpoints include model, normalizers, configuration, and dataset version.

## Acceptance

Report action error, trajectory ADE/FDE, time error, goal error, feasibility metrics, per-mode and jump-strength breakdowns, fair GRU/MLP comparison, hidden reset, save/load, and deterministic inference.

# PI5: Causal Teacher and DAgger

## Goal

Reduce Oracle-unimitability and cover states visited by the learned student.

## Causal Teacher

At minimum compare current/history-truth CV and CA forecasts; an optional learned causal predictor may follow. Use the same intercept-time search and Ego planner against predicted Target trajectories.

## DAgger Loop

1. Run student closed loop.
2. Record visited states.
3. Query causal teacher at those states.
4. Optionally query Oracle for upper-bound analysis.
5. Store separate action and trajectory labels.
6. Merge versioned data.
7. Retrain student.
8. Validate independently.

Conflicting Oracle actions for identical student observations must be labeled as ambiguity, not silently averaged.

## Acceptance

Compare Oracle, causal teacher, truth-history student, damaged-input student, behavior-cloning student, and DAgger student. Report closed-loop success, miss distance, recovery, per-round performance, teacher query count, and label ambiguity rate.

# PI6: PPO Closed-Loop Fine-Tuning

## Goal

Optimize closed-loop crossing under strict student observations, initialized from an accepted PI4/PI5 checkpoint.

## Actor-Critic Boundary

Actor uses only deployable student information. Critic may use current Target/Ego truth, current estimation error/validity, and current dynamics/wind/delay. Critic cannot use future Target truth, future segments, Oracle plans, future wind, or future commands.

Critic dimension and layout require independent definition and tests.

## Training Stages

- Stage A: stronger imitation loss and small PPO updates.
- Stage B: reduce imitation and increase closed-loop terminal objective.
- Stage C: near-zero imitation; optimize true crossing and constraints.

Forbidden rewards include relative-speed convergence, hold, following stability, and Target future-truth distance.

## Acceptance

Report overall/per-mode success, miss distance p50/p95, intercept time, exclusion/workspace/height/speed/acceleration/jerk violations, action rate, jump response, deterministic replay, and BC/DAgger/PPO comparison.

# PI7: Upstream Estimator and Integrated Robustness

## Goal

Integrate actual upstream relative-state estimator outputs or honestly labeled recorded/conservative error distributions and evaluate combined robustness.

## Planned Inputs

- Fisheye detection/tracking.
- Relative direction and range estimates.
- Relative velocity estimate.
- Confidence and validity.

If the real estimator is unavailable, use recordings or conservative models and state that visual closed loop is not complete.

## Combined Disturbances

- Distance jump, drift, and velocity spike.
- Observation delay and dropout.
- Ego dynamics randomization.
- Execution delay.
- Wind and gust.
- Complex Target maneuvers.

## Comparison Matrix

```text
Oracle: complete future truth
Causal teacher: historical truth
Truth-history student: no future
Simulated-estimate student: error model
Visual student: actual upstream output
```

## Acceptance

Report end-to-end success by range, view angle, altitude difference, horizontal distance, confidence, jump magnitude, Target mode, and disturbance strength. Also report miss distance, action physicality, run frequency, inference latency, failure distribution, and an explicit simulation-to-real gap statement.

# Unified Evaluation and Reporting

## Teacher Metrics

- Feasible rate.
- Terminal spatial and timing error p50/p95.
- Solve time p50/p95.
- Constraint margins.
- Infeasible reasons.

## Offline Student Metrics

- Action error.
- Trajectory ADE/FDE.
- Time-to-intercept error.
- Goal error.
- Feasibility metrics.

## Closed-Loop Student Metrics

- Success rate and miss distance.
- Completion time.
- Entity exclusion and physical violations.
- Action discontinuity and jump response.
- Deterministic repeatability.

All metrics include episode count and seed, overall and per Target mode, major disturbance bins, and p50/p95 where applicable.

Formal reports include 3D replay, Target truth, Target estimate history, teacher/student Ego trajectories, virtual-offset trajectory, intercept event, state/command/constraint time series, confidence/age/jump flags, failure reasons, training curves, and comparison tables. Successful videos without batch statistics are insufficient.

# Required Test Categories

- Legacy regression against `m7b-accepted`.
- Oracle-only future permission.
- Student input/normalizer/checkpoint/dataset leakage audit.
- Counterfactual strict-causality test.
- Dataset version/frame/unit/sample-period validation.
- Per-environment hidden-state reset and no crosstalk.
- Continuous high-speed crossing detection.
- Bounded action response to anomalous observations.
- Fixed-seed/checkpoint determinism.

# Current Stop Point

PI0 documentation is the only active work. After PI0 self-tests, stop for independent read-only review. PI1 may begin only after PI0 independent review, user acceptance, and separate explicit PI1 authorization.
