# Predictive Intercept Task Definition

## Scope

Predictive Intercept is a new non-contact virtual-offset crossing task. It is independent from the frozen Legacy offset-rendezvous tasks and must use independent future task IDs, environments, rewards, terminations, checkpoints, and evaluation reports.

This document freezes task semantics only. PI0 does not implement an environment.

## Research Objective

Ego starts at ground-adjacent or low altitude while Target is already airborne. Using imperfect, strictly causal Target state estimates, Ego must infer useful future motion and intercept timing, then cross the moving virtual-offset trajectory at a feasible time.

The objective is a one-time spatiotemporal crossing. It is not prolonged following, formation keeping, contact, or a requirement to match Target velocity.

## Coordinate Definitions

The first version uses each environment's local world-aligned frame `_w`.

```text
p_rel_w(t) = p_target_w(t) - p_ego_w(t)
v_rel_w(t) = v_target_w(t) - v_ego_w(t)
g_w(t) = p_target_w(t) + b_des_w
```

`g_w(t)` is the virtual non-contact trajectory associated with Target.

## Mandatory Nonzero Offset

The configured offset must satisfy:

```text
||b_des_w|| >= b_min > 0
```

Zero offset is invalid at configuration validation and runtime. Target's geometry center must never be used as the terminal goal.

Geometry must also satisfy:

```text
||b_des_w|| > d_exclusion + r_tol + m_geometry
```

This prevents overlap between the virtual crossing tolerance and the Target entity exclusion zone.

## Initial Scenario

Ego starts near ground level with near-zero velocity:

```text
z_ego_0 ~= z_ground + h_init
||v_ego_0|| ~= 0
```

Target starts airborne with nonzero horizontal separation in typical episodes:

```text
z_target_0 > z_ego_0
d_xy_0 > 0
```

The representative maneuver is an oblique climbing intercept with horizontal lead, not a degenerate vertical ascent beneath a horizontal constant-velocity Target.

At minimum, future training and validation must randomize:

- Target initial altitude.
- Ego-Target horizontal distance and bearing.
- Target initial speed magnitude and direction.
- Target motion mode and maneuver parameters.
- Ego initial heading.
- Nonzero offset direction and magnitude, unless separate fixed-offset policies are deliberately evaluated.

Before high-fidelity dynamics are authorized, "ground takeoff" means near-ground initialization plus high-level 3D velocity commands through simplified velocity dynamics. It does not claim motor startup, ground effect, landing-gear contact, attitude-loop fidelity, or physical liftoff dynamics.

## Intercept Event

An Oracle teacher may select an intercept time `T_star > t_0`. The ideal condition is:

```text
p_ego_w(T_star) = p_target_w(T_star) + b_des_w
```

The numerical tolerance is:

```text
||p_ego_w(T_star) - g_w(T_star)|| <= r_tol
```

Success does not require:

- `v_ego_w(T_star) = v_target_w(T_star)`.
- Relative velocity approaching zero.
- Hovering at the crossing point.
- A hold interval.
- Following Target after crossing.

The first valid crossing immediately terminates the episode successfully.

## Continuous Crossing Detection

Both Ego and the virtual offset move during a simulation step. Define:

```text
r_0 = p_ego_w,k - g_w,k
r_1 = p_ego_w,k+1 - g_w,k+1
delta_r = r_1 - r_0
```

Under linear relative motion within the step:

```text
r(lambda) = r_0 + lambda * delta_r, lambda in [0, 1]
```

For `||delta_r||^2 > epsilon`, compute:

```text
lambda_star = clamp(-dot(r_0, delta_r) / ||delta_r||^2, 0, 1)
d_min_step = ||r_0 + lambda_star * delta_r||
```

For negligible `delta_r`, use `d_min_step = ||r_0||`. A virtual crossing occurs when:

```text
d_min_step <= r_tol
```

Future implementation tests must cover high-speed crossings that endpoint-only tests would miss.

## Target Entity Exclusion

Entity distance is independent from virtual crossing distance:

```text
d_entity = ||p_ego_w - p_target_w||
```

If `d_entity < d_exclusion`, terminate with `target_exclusion_violation`.

If virtual crossing and entity exclusion happen in the same step, entity exclusion takes precedence and success must not be recorded.

Target contact, impact, or center crossing is never success.

## Initial Control Interface

The first authorized implementation stage is planned to retain a high-level 3D world-frame velocity command:

```text
a_raw in R^3
v_cmd_w = v_max * tanh(a_raw)
```

This is an interface planning decision, not PI1 authorization. The environment owns command mapping, dynamic response, and physical limits.

## Failure Taxonomy

Future environments must record separate reasons:

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

A single undifferentiated `failure` field is insufficient.

## Evaluation Contract

Closed-loop reports must include episode count and seed plus overall and per-motion-mode values for:

- Success rate.
- Miss distance p50/p95.
- Completion or intercept time p50/p95.
- Entity exclusion violation rate.
- Workspace, height, speed, acceleration, and jerk violation rates.
- Action discontinuity and jump response.
- Deterministic repeatability.

Major disturbance strengths must be binned. Successful videos alone are not acceptance evidence.

## Legacy Separation

Predictive Intercept must not reuse Legacy task IDs or claim Legacy checkpoints solve this task. Legacy rewards for relative-speed matching, hold duration, and offset following must not be restored in the final autonomous Predictive Intercept objective.
