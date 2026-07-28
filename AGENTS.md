# Project Rules

These rules are mandatory for all work in `/home/lab_726/uav_rendezvous_rl`.

## Environment

1. Run all Isaac Lab commands through:

   ```bash
   /home/lab_726/IsaacLab/isaaclab.sh
   ```

2. Do not use Anaconda Python.
3. Before running Isaac Lab, clear `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `VIRTUAL_ENV`, `PYTHONPATH`, and `PYTHONHOME`.
4. Do not modify `/home/lab_726/IsaacLab`, `/home/lab_726/isaacsim`, Pegasus Simulator, system Python dependencies, or the NVIDIA driver.
5. Do not execute `sudo`.

## Current Authorization

6. The current program is Predictive Intercept and the only authorized milestone is PI0.
7. PI0 is governance and design documentation only. PI1 through PI7 are described for interface closure and audit planning but are not authorized for implementation.
8. PI0 may modify only:

   ```text
   AGENTS.md
   README.md
   docs/milestone_state.md
   ```

9. PI0 may create only:

   ```text
   docs/repository_reconciliation.md
   docs/predictive_intercept_task_definition.md
   docs/teacher_student_architecture.md
   docs/predictive_intercept_information_boundary.md
   docs/predictive_intercept_dataset_schema.md
   docs/predictive_intercept_milestones.md
   ```

10. PI0 must not modify `source/`, create `source/uav_predictive_intercept/`, implement empty PI1 interfaces, create training scripts, run training, or generate a formal dataset.
11. Do not enter PI1, create a PI0 commit, merge, tag, or push PI0 changes without explicit user authorization after independent PI0 review.

## Legacy Freeze

12. The authoritative Legacy freeze baseline is:

   ```text
   tag: m7b-accepted
   commit: fc2cb17315e7f6b2b8dee463db5ab73f931c938e
   ```

13. `m7a-accepted` at `555daeb598ef633f7e2dbf8b86a12148cd48cf34` remains an M7A historical acceptance baseline, not the final Predictive Intercept Legacy baseline.
14. The frozen Legacy task IDs are:

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

15. Do not modify the frozen Legacy tasks' rewards, success criteria, terminations, Actor/Critic observations, action spaces, task IDs, checkpoint meanings, or acceptance metrics.
16. Do not rename or migrate `source/uav_rendezvous_rl/`. Predictive Intercept uses the separately planned package name `uav_predictive_intercept` only after PI1 authorization.

## Predictive Intercept Safety

17. Predictive Intercept is a non-contact virtual-offset crossing task. The offset must satisfy `||b_des_w|| >= b_min > 0`; zero offset is forbidden.
18. Target entity contact is never success. The Target entity hard exclusion zone must remain independent of virtual-offset crossing success.
19. The geometry contract must enforce `||b_des_w|| > d_exclusion + r_tol + m_geometry` before any future implementation is accepted.
20. If virtual crossing and Target exclusion occur in the same step, exclusion failure takes precedence.

## Information Boundaries

21. Only the Oracle teacher may read complete future Target truth, future maneuver switches, or complete future trajectories.
22. The causal teacher may read current and historical truth plus current dynamics/disturbance state, but no future Target or disturbance information.
23. The student may read only strictly causal deployable estimates, history, confidence/validity metadata, measurable Ego state, nonzero offset configuration, and previous or executed action state.
24. Student inference must never receive Target future truth, motion mode labels, generator parameters, future segment schedules, future commands, Oracle intercept time, or Oracle planned trajectory.
25. Teacher future information may be stored as a supervised label in a versioned dataset but must never be included in student input fields or normalization statistics.

## Coordinate Definitions

26. World-aligned quantities use suffix `_w`. Fixed definitions are:

   ```text
   p_rel_w = p_target_w - p_ego_w
   v_rel_w = v_target_w - v_ego_w
   g_w = p_target_w + b_des_w
   ```

## Workflow

27. Before each work session, reread `AGENTS.md`, `docs/milestone_state.md`, and `docs/predictive_intercept_milestones.md`, plus the latest independent audit report.
28. Every milestone requires read-only pre-audit, user scope confirmation, implementation, independent read-only review, user acceptance, and a separate user decision for commit/merge/next-stage entry.
29. Implementation self-tests are not independent acceptance.
30. After PI0 documentation is complete, stop and wait for independent PI0 review. PI1 remains unauthorized.
