# UAV Rendezvous RL

External Isaac Lab project for a staged non-contact UAV offset rendezvous task.

M3 has been completed and passed user technical acceptance at tag `m3-accepted`.

M3 provides a vectorized target motion library for the dual-placeholder truth environment. The implemented target modes are `ConstantVelocity`, `ConstantAcceleration`, `ConstantTurn`, and `PiecewiseAcceleration`. The task tensors remain the single source of truth; `RigidObject` placeholders are synchronized visualization/state carriers.

Actor observations remain limited to `[p_rel_w, v_rel_w]`. They do not expose target motion mode labels, generator parameters, future schedules, or future target states.

M4 adds an independent deterministic non-learning offset rendezvous baseline task. M4 has passed user technical acceptance. The original Direct task remains the M2/M3 regression task with stationary ego and no-op actions.

The M4 baseline controller uses only current deployable state: `p_ego_w`, `v_ego_w`, `p_target_w`, `v_target_w`, and fixed `b_des_w`. It does not read target motion mode, generator parameters, future schedules, or future target states.

M2, M3, and M4 runtime audits have passed locally. M4 is accepted.

M5 is authorized but not started, and is not implemented in this repository state.

## Task

```text
Isaac-Uav-Rendezvous-Direct-v0
Isaac-Uav-Rendezvous-Baseline-v0
```

## Isaac Lab Entry

All Isaac Lab commands must run through:

```bash
/home/lab_726/IsaacLab/isaaclab.sh
```

Clear Conda, virtualenv, ROS Python paths, and Python home before launching Isaac Lab:

```bash
env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u VIRTUAL_ENV \
  /home/lab_726/IsaacLab/isaaclab.sh -p <script-or-module>
```

## Smoke Commands

Install editable package:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pip install -e /home/lab_726/uav_rendezvous_rl/source/uav_rendezvous_rl
```

Run tests:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p -m pytest /home/lab_726/uav_rendezvous_rl/tests -q
```

Run zero/random agents:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/zero_agent.py --task Isaac-Uav-Rendezvous-Direct-v0 --num_envs 16 --device cuda:0 --headless --steps 10000
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/random_agent.py --task Isaac-Uav-Rendezvous-Direct-v0 --num_envs 16 --device cuda:0 --headless --steps 10000
```

Run M2 runtime audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m2_runtime.py --num_envs 16 --steps 1000 --seed 42 --device cuda:0 --headless
```

Run M3 runtime audit:

```bash
env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u VIRTUAL_ENV \
  /home/lab_726/IsaacLab/isaaclab.sh -p \
  /home/lab_726/uav_rendezvous_rl/scripts/audit_m3_motion_runtime.py \
  --num_envs 16 \
  --steps 5000 \
  --seed 42 \
  --split train \
  --device cuda:0 \
  --headless
```

Run M4 baseline runtime audit:

```bash
env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u VIRTUAL_ENV \
  /home/lab_726/IsaacLab/isaaclab.sh -p \
  /home/lab_726/uav_rendezvous_rl/scripts/audit_m4_baseline_runtime.py \
  --num_envs 64 \
  --episodes 5 \
  --seed 42 \
  --split train \
  --device cuda:0 \
  --headless
```
