# Teacher-Student Architecture

## Status

This PI0 document freezes architectural roles and future interfaces. It does not implement modules, create package directories, train networks, or authorize PI1.

## System Flow

```text
Target motion generator
    |-- complete future truth ----------------------> Oracle teacher
    |                                                   |
    |                                        T_star and feasible Ego plan
    |                                                   |
    |                                           supervised labels
    |
    |-- current and historical truth
            |
            v
      estimation/error model or upstream estimator
            |
            v
      causal estimates + confidence + validity + history
            |
            v
      temporal student policy
            |
            |-- current action
            |-- short-horizon reference head
            `-- auxiliary predictions
            |
            v
      3D world velocity command
            |
            v
      simplified or later high-fidelity Ego dynamics
```

## Oracle Teacher

The Oracle teacher is a non-neural planning baseline. It may read:

- Current and complete future Target truth.
- Future Target maneuver switches and segment schedules.
- Current Ego truth and dynamics constraints.
- Current nonzero offset configuration.
- Current-stage dynamics and disturbance truth when explicitly allowed.

It jointly selects:

- Feasible intercept time `T_star`.
- Timed Ego reference position, velocity, acceleration, and jerk.
- Reference action or velocity command.
- Feasibility status, trajectory cost, and constraint margins.

Oracle outputs are used only for upper bounds, supervised labels, trajectory-quality references, and control-problem isolation. They are never student inference inputs.

## Oracle Planning Method

The first planned method is deliberately auditable:

```text
candidate intercept-time grid
    + analytic trajectory generation
    + actual Ego-dynamics rollout
    + feasibility checks
    + cost selection
```

For candidate `T_i = t_0 + tau_i`, form:

```text
g_i = p_target_w(T_i) + b_des_w
```

Generate a quintic trajectory per axis satisfying current position, velocity, and acceleration plus terminal position `g_i`. Terminal velocity and acceleration are either free with regularization or use a loose reference; Target velocity matching is not mandatory.

Derive `v_ref`, `a_ref`, and `j_ref`. For the initial simplified first-order velocity response, map acceleration to a reference command approximately as:

```text
v_cmd_ref ~= v_ego_w + tau_v * a_ref_w
```

Analytic feasibility alone is insufficient. Every candidate must be re-rolled through the actual authorized environment dynamics and checked for speed, acceleration, jerk, workspace, height, entity exclusion, finite state, action interface, and terminal error.

A first candidate cost may include time, terminal spatial error, terminal speed regularization, integrated acceleration, integrated jerk, and command-rate penalties. The minimum-cost feasible candidate is selected. If none is feasible, return a stable `infeasible` result and explicit reason.

Direct shooting, collocation, nonlinear MPC, MINCO, and other complex optimizers are comparisons only after the analytic baseline is accepted.

## Causal Privileged Teacher

The causal teacher may read:

- Current and historical Target truth.
- Current Ego truth.
- Current dynamics and disturbance state.

It may not read future Target truth, future maneuver switches, future commands, complete future trajectories, or future disturbances. It must causally predict Target motion and plan against that prediction.

Its future role is to provide more imitable action labels, support DAgger, and measure performance lost due to intrinsically unavailable future information.

## Student

The student reads only strictly causal deployable information:

- Estimated relative position and velocity.
- Causal observation history.
- Measurable Ego state.
- Nonzero offset configuration.
- Position/velocity confidence or uncertainty.
- Measurement validity, observation age, and jump/innovation metadata.
- Previous policy-issued or executed action state as explicitly frozen later.

The student cannot read Target future truth, motion mode, generator parameters, future segment schedules, future commands, Oracle intercept time, or Oracle trajectories.

## Student Network Plan

The primary first student is a single-layer GRU with initial hidden size 128. A stacked-history MLP is the required fair control.

```text
per-frame encoder
    -> GRU or stacked-history MLP
    -> latent state
        |-- 3D current action head
        |-- H x 3 short-horizon reference-velocity head
        `-- time/goal/feasibility auxiliary heads
```

The action head emits `a_raw in R^3`, mapped by the environment to a bounded velocity command. The short-horizon head emits future Ego reference velocities or relative displacements; reference velocity is preferred initially because it matches the control interface. Deployment executes only the first command and replans next cycle.

Auxiliary predictions include time-to-intercept, teacher intercept point relative to Ego, feasibility, and optional uncertainty. They are training outputs, not future-truth inputs fed back into the action head.

## Estimation Protection

Distance jumps can create unrealistic finite-difference velocity spikes. Future estimation infrastructure must combine:

- Innovation gating.
- Finite-state correction.
- Confidence reduction.
- Velocity filtering.
- Observation age.
- Valid masks.
- Jump flags.
- Action continuity and physical speed/acceleration/jerk limits.

PPO alone is not an acceptable substitute for estimator protection.

## Training Sequence

1. Oracle feasibility and execution baseline.
2. Versioned, replayable teacher dataset.
3. Strictly causal damaged student observations.
4. Behavior cloning with action, trajectory, time, goal, and feasibility losses.
5. Causal teacher and DAgger on student-visited states.
6. PPO fine-tuning with current-time privileged Critic information only.
7. Integration with real upstream estimator outputs or honestly labeled conservative recordings/models.

## PPO Boundary

The Actor uses only student-deployable fields and its own causal memory. The Critic may use current Target/Ego truth, current estimation error and validity, and current dynamics/wind/delay state. It may not use future Target truth, future segments, Oracle future plans, future wind, or future commands.

## Planned Package Layout

After PI1 authorization, the separate package may contain:

```text
uav_predictive_intercept/
    tasks/direct/
    teacher/
    student/
    estimation/
    imitation/
    rewards/
    terminations/
    metrics/
    visualization/
    common/
```

PI0 creates none of these directories.
