# Environment Audit

Milestone: 0
Date: 2026-07-21
Workspace: `/home/lab_726/uav_rendezvous_rl`

## Scope

This audit was performed as a read-only inspection of the existing Ubuntu, Isaac Sim, Isaac Lab, RSL-RL, and project workspace state. No Isaac Sim, Isaac Lab, Pegasus, system package, or Python dependency source files were modified during the audit.

## Host And GPU

| Item | Observed value |
| --- | --- |
| Operating system | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 5090 D v2 |
| NVIDIA driver | 580.173.02 |
| GPU memory | 24455 MiB |

Command used:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

## Isaac Sim

| Item | Observed value |
| --- | --- |
| Isaac Sim path requested by user | `/home/lab_726/isaacsim/isaacsim/_build/linux-x86_64/release` |
| Isaac Lab `_isaac_sim` symlink target | `/home/lab_726/isaacsim/isaacsim/_build/linux-x86_64/release` |
| Isaac Sim VERSION file | `5.1.0-rc.19+main.0.aa503a9b.local` |
| Python launcher | `/home/lab_726/IsaacLab/_isaac_sim/python.sh` |
| Kit Python executable | `/home/lab_726/IsaacLab/_isaac_sim/kit/python/bin/python3` |
| Kit Python version | `3.11.13` |

Files checked:

```text
/home/lab_726/IsaacLab/_isaac_sim/VERSION
/home/lab_726/IsaacLab/_isaac_sim/python.sh
/home/lab_726/IsaacLab/_isaac_sim/setup_python_env.sh
```

Commands used:

```bash
readlink -f /home/lab_726/IsaacLab/_isaac_sim
/home/lab_726/IsaacLab/isaaclab.sh -p -c "import sys; print(sys.executable); print(sys.version)"
```

## Isaac Lab

| Item | Observed value |
| --- | --- |
| Isaac Lab path | `/home/lab_726/IsaacLab` |
| Isaac Lab VERSION | `2.3.2` |
| Git description | `v2.3.2` |
| Git state | detached HEAD, no dirty files reported by `git status --short --branch` |
| Recent commit | `37ddf62 Bumps version to v2.3.2 (#4399)` |

Files checked:

```text
/home/lab_726/IsaacLab/VERSION
/home/lab_726/IsaacLab/pyproject.toml
/home/lab_726/IsaacLab/environment.yml
/home/lab_726/IsaacLab/isaaclab.sh
```

Commands used:

```bash
git status --short --branch
git describe --tags --always --dirty
git log --oneline -10
```

## Python And RL Dependencies

The audit used Isaac Lab's launcher, not Anaconda Python:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lab_726/IsaacLab/isaaclab.sh -p -c "..."
```

| Package | Observed value |
| --- | --- |
| `torch` | `2.7.0+cu128` |
| CUDA available to Torch | `True` |
| Torch CUDA version | `12.8` |
| CUDA device count | `1` |
| CUDA device 0 | `NVIDIA GeForce RTX 5090 D v2` |
| `rsl-rl-lib` | `3.1.2` |
| `gymnasium` | `1.2.1` |
| `numpy` | `1.26.0` |
| `isaaclab` | `0.54.2` |
| `isaaclab_rl` | `0.4.7` |
| `isaaclab_tasks` | `0.11.12` |
| `isaaclab_assets` | `0.2.4` |
| `isaaclab_contrib` | `0.0.2` |

`pip check` reported existing dependency warnings in the Isaac Sim Kit environment:

```text
nvidia-srl-usd-to-urdf 1.0.2 requires usd-core, which is not installed.
nvidia-srl-base 1.3.0 requires docstring-parser, which is not installed.
nvidia-srl-usd 2.0.0 requires usd-core, which is not installed.
msal 1.27.0 requires pyjwt, which is not installed.
nvidia-srl-usd-to-urdf 1.0.2 has requirement lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0.
fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.
```

No dependency installation, upgrade, or removal was performed.

## Conda And Environment Variables

Observed values at audit time:

| Variable | Observed value |
| --- | --- |
| `CONDA_PREFIX` | unset |
| `CONDA_DEFAULT_ENV` | unset |
| `VIRTUAL_ENV` | unset |
| `PYTHONPATH` | `/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages` |

Risk: ROS Humble Python 3.10 paths are present in `PYTHONPATH`. They must not be passed into Isaac Sim Kit Python for this project. All Isaac Lab commands for later milestones should run with ROS-related `PYTHONPATH` cleared, for example:

```bash
env -u PYTHONPATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV /home/lab_726/IsaacLab/isaaclab.sh -p <script-or-module>
```

## External Project Template

The Isaac Lab external project template exists under:

```text
/home/lab_726/IsaacLab/tools/template
```

Relevant files checked:

```text
/home/lab_726/IsaacLab/tools/template/templates/external/README.md
/home/lab_726/IsaacLab/tools/template/generator.py
/home/lab_726/IsaacLab/tools/template/cli.py
/home/lab_726/IsaacLab/tools/template/templates/extension/setup.py
/home/lab_726/IsaacLab/tools/template/templates/extension/pyproject.toml
/home/lab_726/IsaacLab/tools/template/templates/tasks/direct_single-agent/env
/home/lab_726/IsaacLab/tools/template/templates/tasks/direct_single-agent/env_cfg
/home/lab_726/IsaacLab/tools/template/templates/tasks/__init__task
/home/lab_726/IsaacLab/tools/template/templates/agents/rsl_rl_ppo_cfg
```

Findings:

| Area | Finding |
| --- | --- |
| External layout | Template creates an isolated project outside Isaac Lab under `source/<package>` with `setup.py`, `pyproject.toml`, `config/extension.toml`, `tasks/`, and scripts. |
| Editable install | Template README recommends `python -m pip install -e source/<package>` or Isaac Lab launcher equivalent. |
| Gym registration | Task registration is done with `gym.register(...)` in the task package `__init__.py`. |
| Direct RL template | Direct workflow implements `DirectRLEnv` with `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, and `_reset_idx`. |
| RSL-RL config | Template provides `RslRlOnPolicyRunnerCfg`, `RslRlPpoActorCriticCfg`, and `RslRlPpoAlgorithmCfg`. |

## DirectRLEnv Structure

Relevant files checked:

```text
/home/lab_726/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py
/home/lab_726/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py
/home/lab_726/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
/home/lab_726/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py
```

Important DirectRLEnv behavior:

| Function or config | Behavior |
| --- | --- |
| `DirectRLEnvCfg.decimation` | Number of physics steps per environment/policy step. |
| `DirectRLEnvCfg.sim.dt` | Physics time step. |
| `step_dt` | `decimation * sim.dt`. |
| `_pre_physics_step(actions)` | Called once per policy step before physics substeps. |
| `_apply_action()` | Called every physics substep. |
| `_get_observations()` | Returns observation dictionary, normally `{"policy": tensor}` and optionally `{"critic": tensor}` for asymmetric actor-critic. |
| `_get_dones()` | Returns `(terminated, time_out)`. |
| `_reset_idx(env_ids)` | Must restore per-env simulation and custom buffers. |
| `state_space` | Enables critic observation space for asymmetric actor-critic. |

## RSL-RL And Recurrent PPO

Relevant files checked:

```text
/home/lab_726/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py
/home/lab_726/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py
/home/lab_726/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/agents/rsl_rl_ppo_cfg.py
/home/lab_726/IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/modules/actor_critic_recurrent.py
/home/lab_726/IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/networks/memory.py
/home/lab_726/IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py
```

Findings:

| Area | Finding |
| --- | --- |
| Recurrent config | Isaac Lab exposes `RslRlPpoActorCriticRecurrentCfg` with `rnn_type`, `rnn_hidden_dim`, and `rnn_num_layers`. |
| GRU support | RSL-RL `Memory` uses `nn.GRU` when `rnn_type="gru"`; otherwise it uses LSTM. |
| Hidden reset | RSL-RL calls `policy.reset(dones)` after each environment step; done masks zero hidden states per environment. |
| Asymmetric critic | `obs_groups={"policy": ["policy"], "critic": ["critic"]}` maps actor and critic observations separately. |
| Normalization | Actor and critic observation normalization are configured separately. |
| Checkpoints | `OnPolicyRunner.save()` saves model state, optimizer state, current iteration, and optional infos. |

## UAV And Multirotor Assets

Relevant files checked:

```text
/home/lab_726/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py
/home/lab_726/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/agents/rsl_rl_ppo_cfg.py
/home/lab_726/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/quadcopter.py
/home/lab_726/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/arl_robot_1.py
/home/lab_726/IsaacLab/source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor.py
/home/lab_726/IsaacLab/source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor_cfg.py
/home/lab_726/IsaacLab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py
/home/lab_726/IsaacLab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster_cfg.py
```

Findings:

| Asset or interface | Finding |
| --- | --- |
| `Isaac-Quadcopter-Direct-v0` | Direct RL quadcopter example exists and applies total thrust plus body moment to a Crazyflie articulation. |
| `CRAZYFLIE_CFG` | Available under `isaaclab_assets.robots.quadcopter`, uses USD path `Robots/Bitcraze/Crazyflie/cf2x.usd`. |
| `MultirotorCfg` | Available in `isaaclab_contrib`, extends `ArticulationCfg` with thrusters and allocation matrix. |
| `ThrusterCfg` | Supports thrust range, thrust constant range, rise/fall time constants, torque-to-thrust ratio, and per-thruster names. |
| `Multirotor.set_thrust_target` | Sets per-thruster thrust targets. |
| `Multirotor.write_data_to_sim` | Applies actuator model and combined wrench at physics step. |
| `RigidObject` | Provides root pose/velocity writers and external wrench interfaces suitable for simplified M1-M5 assets. |

Recommendation: do not introduce Pegasus, PX4, ROS 2, Crazyflie, or Multirotor/Thruster in M1-M4. Use simplified `RigidObject`/sphere placeholders through M4 and first train V0 on simplified velocity dynamics in M5. Introduce Crazyflie or `Multirotor/Thruster` only after V0 is stable.

## Project Directory State

Observed project path:

```text
/home/lab_726/uav_rendezvous_rl
```

State before writing milestone 0 documents:

| Item | Observed value |
| --- | --- |
| Directory exists | yes |
| Contents before docs | empty |
| Git repository | no |

Command used:

```bash
```

Result: not a Git repository.

## Audit Risks And Notes

1. Isaac Lab scripts must be run with Conda deactivated.
2. ROS-related `PYTHONPATH` must be cleared when launching Isaac Sim Kit Python.
3. Direct imports of some Isaac Lab modules without launching Isaac Sim/AppLauncher can fail because Omniverse modules such as `pxr` are not initialized in the same way as runtime scripts.
4. `pip check` reports existing dependency warnings; no dependency changes are planned for this project stage.
5. RSL-RL recurrent hidden-state reset depends on correct `dones`; environment termination and timeout masks must be correct before training.
6. Safety distance `d_safe=0.75 m` is valid only for the placeholder spheres with `r_ego=0.20 m`, `r_target=0.20 m`, and `safety_margin=0.35 m`. It must be recalculated for real UAV collision geometry.
