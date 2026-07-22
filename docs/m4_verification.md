# M4 Verification

Milestone: M4
Status: completed_pending_user_acceptance
Date: 2026-07-22
Branch: `feature/m4`

## Scope

M4 implements a deterministic, non-learning, non-contact offset rendezvous baseline in an independent task, `Isaac-Uav-Rendezvous-Baseline-v0`. The original `Isaac-Uav-Rendezvous-Direct-v0` task remains the M2/M3 regression task with stationary ego and no-op actions.

M4 did not enter M5. No PPO training, Actor/Critic networks, recurrent policies, rewards for learning, B-splines, Crazyflie, Multirotor/Thruster, Pegasus, PX4, ROS 2, or perception network work was performed.

## Runtime Command Requirements

All Isaac Lab commands were run through `/home/lab_726/IsaacLab/isaaclab.sh` with Conda, virtualenv, ROS Python paths, and Python home cleared:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p <script-or-module>
```

## Commands Run

Syntax check:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m compileall -q scripts source tests
```

Pytest:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pytest /home/lab_726/uav_rendezvous_rl/tests -q
```

M4 small smoke audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m4_baseline_runtime.py --num_envs 8 --episodes 1 --seed 42 --split train --device cuda:0 --headless
```

M2 regression audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42 --device cuda:0 --headless
```

M3 regression audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m3_motion_runtime.py --num_envs 16 --steps 5000 --seed 42 --split train --device cuda:0 --headless
```

Formal M4 baseline audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m4_baseline_runtime.py --num_envs 64 --episodes 5 --seed 42 --split train --device cuda:0 --headless
```

Direct zero-agent regression:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/zero_agent.py --task Isaac-Uav-Rendezvous-Direct-v0 --num_envs 16 --device cuda:0 --headless --steps 10000
```

Direct random-agent regression:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/random_agent.py --task Isaac-Uav-Rendezvous-Direct-v0 --num_envs 16 --device cuda:0 --headless --steps 10000
```

Post-documentation pytest:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pytest /home/lab_726/uav_rendezvous_rl/tests -q
```

## Results

Pytest passed:

```text
41 passed
```

M2 audit passed with finite truth state, static ego, target motion, synchronized assets, and no contact. Reported minimum relative distance was `4.625568866729736 m` with `d_safe = 0.75 m`.

M3 audit passed for `5000` steps with all four target motion modes covered, actor observation isolation intact, finite state, synchronized assets, no ego drift, and no collision risk. Reported all-physics-substep minimum relative distance was `4.436470985412598 m` with `d_safe = 0.75 m`.

M4 small smoke audit passed for `8` envs and `1` episode per scenario.

Formal M4 audit passed for `64` envs x `5` episodes per scenario. Acceptance checks all returned true.

Nominal fixed ConstantVelocity metrics:

| Metric | Value |
| --- | ---: |
| Success rate | `1.0` |
| Collision count | `0` |
| Success offset error p95 | `7.62939453125e-06 m` |
| Success relative speed p95 | `8.58306884765625e-06 m/s` |
| Target analytic error | `9.5367431640625e-07 m` |

Random ConstantVelocity metrics:

| Metric | Value |
| --- | ---: |
| Success rate | `1.0` |
| Collision count | `0` |
| Success offset error p95 | `4.206398443784565e-05 m` |
| Success relative speed p95 | `8.18166954559274e-05 m/s` |
| Target analytic error | `3.814697265625e-06 m` |

Stress-mode metrics from the formal M4 audit:

| Scenario | Success rate | Collision count | Success offset error p95 | Success relative speed p95 |
| --- | ---: | ---: | ---: | ---: |
| ConstantAcceleration | `1.0` | `0` | `0.0022186809219419956 m` | `7.15116475475952e-05 m/s` |
| ConstantTurn | `1.0` | `0` | `0.009585591964423656 m` | `0.0008776930626481771 m/s` |
| PiecewiseAcceleration | `1.0` | `0` | `0.00256736995652318 m` | `0.004727173130959272 m/s` |

Direct zero-agent 10000-step regression passed on `Isaac-Uav-Rendezvous-Direct-v0` with finite diagnostics, static ego, and target motion.

Direct random-agent 10000-step regression passed on `Isaac-Uav-Rendezvous-Direct-v0` with finite diagnostics, static ego, and target motion.

## Warnings Observed

The Isaac Lab/Isaac Sim runtime emitted warnings during headless audits. They did not fail the audits and no external dependencies or system components were modified.

- PhysX warning: `enable_external_forces_every_iteration` is set to `False`.
- Omniverse warning: enable `omni.materialx.libs` extension to use MaterialX.
- Carb warning: non-optional `omni::physx::IPhysxBenchmarks` plugin interface not listed as dependency.
- Deprecation warning: `omni.isaac.dynamic_control` is deprecated as of Isaac Sim 4.5.
- GPU foundation warnings: unsupported Intel integrated GPU skipped, CPU performance profile set to powersave, IOMMU enabled.
- Omni Fabric warning: `/World/envs/env_0:omni:rtx:skip` has no valid data.
- USD shutdown warning: unexpected reference count while closing anonymous `World0.usd` stage.

## M4 Limitations

M4 uses exact target truth, simplified ego kinematics, and short-horizon current-state extrapolation. Low error under the audited ConstantVelocity target does not represent final performance under perception noise, observation delay, disturbances, aggressive or complex maneuvers, or real multirotor dynamics.

M5 remains unauthorized and unimplemented.
