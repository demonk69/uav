# UAV Rendezvous RL

External Isaac Lab project for a staged non-contact UAV offset rendezvous task.

M2 has been implemented and is awaiting user acceptance. Current authorized work remains M2 closure only; M3 is not authorized.

M2 provides dual `ego` and `target` placeholders with truth-state relative observations, fixed-height constant-velocity target motion, and a stationary ego. The task tensors are the single source of truth; `RigidObject` placeholders are synchronized visualization/state carriers. It intentionally does not implement the M3 motion library, deterministic baseline, PPO training, recurrent policy, Pegasus, PX4, ROS 2, Crazyflie, or Multirotor/Thruster dynamics.

## Task

```text
Isaac-Uav-Rendezvous-Direct-v0
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

## M2 Smoke Commands

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
