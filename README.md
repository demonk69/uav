# UAV Rendezvous RL

External Isaac Lab project for a staged non-contact UAV offset rendezvous task.

M4 has passed user technical acceptance at tag `m4-accepted`. The original Direct task remains the M2/M3 regression task with stationary ego and no-op actions.

M3 provides a vectorized target motion library for the dual-placeholder truth environment. The implemented target modes are `ConstantVelocity`, `ConstantAcceleration`, `ConstantTurn`, and `PiecewiseAcceleration`. The task tensors remain the single source of truth; `RigidObject` placeholders are synchronized visualization/state carriers.

M4 adds an independent deterministic non-learning offset rendezvous baseline task. The M4 baseline controller uses only current deployable state: `p_ego_w`, `v_ego_w`, `p_target_w`, `v_target_w`, and fixed `b_des_w`.

M5 adds an accepted independent feedforward RL task, `Isaac-Uav-Rendezvous-RL-v0`, with a 3D raw velocity action mapped by `v_cmd_w = v_max * tanh(a_raw)`, 25D deployable Actor observations, 57D privileged Critic observations, separated reward terms, and explicit safety/workspace/height/speed/non-finite terminations.

M5 has passed user acceptance with independent audit result `ACCEPT M5 WITH NON-BLOCKING ISSUES`. Final M5 verification details are in `docs/m5_verification.md`; the archived audit is in `docs/m5_independent_audit.md`.

M6 implementation is complete and pending user acceptance. It adds independent recurrent task `Isaac-Uav-Rendezvous-Recurrent-v0`, a GRU `ActorCriticRecurrent` PPO config, recurrent-safe play/evaluate reset handling, hidden-state audits, a mixed four-mode target-motion distribution, checkpoint-resume verification, and a fair feedforward ablation task over the same environment. Final M6 verification details are in `docs/m6_verification.md`. The mixed-mode GRU passed absolute safety/performance validation, but did not outperform the fair feedforward ablation.

## M5 Acceptance Summary

- Actor observation: 25D deployable current-state observation.
- Critic observation: 57D privileged current-state observation under the separate `critic` group.
- Action: 3D raw action mapped by `tanh` to a velocity command.
- Policy: feedforward PPO `ActorCritic`; no GRU, LSTM, or recurrent policy is present.
- Final independent validation: validation split, seed `4242`, 64 environments, 4 episodes per environment, 256 total episodes.
- Validation success rate: `100%`.
- Validation collision-risk rate: `0`.
- Successful-episode offset error p95: about `0.36 m`.
- Successful-episode relative speed p95: about `0.17 m/s`.
- Final checkpoint is local only and not tracked by Git: `logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_19-04-26_m5_rewardfix_300_seed42/model_299.pt`.

## M6 Verification Summary

- Recurrent task: `Isaac-Uav-Rendezvous-Recurrent-v0`.
- Fair feedforward ablation task: `Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0`.
- Actor observation: 25D deployable current-state observation; no mode labels, motion parameters, future target states, future commands, or future trajectories.
- Critic observation: 57D privileged current-state observation under the separate `critic` group.
- Mixed GRU validation: validation split, seed `4242`, 64 environments, 8 episodes per environment, balanced 128 episodes per mode; success rate `0.9980`, collision-risk rate `0`, successful offset p95 about `0.463 m`, successful relative-speed p95 about `0.220 m/s`.
- Fair feedforward ablation validation with the same protocol: success rate `1.0`, collision-risk rate `0`, successful offset p95 about `0.198 m`, successful relative-speed p95 about `0.109 m/s`.
- Final mixed GRU checkpoint is local only and not tracked by Git: `logs/rsl_rl/uav_rendezvous_m6_gru/2026-07-22_23-56-05_m6_mixed_gru_300_seed42/model_299.pt`.
- Final feedforward ablation checkpoint is local only and not tracked by Git: `logs/rsl_rl/uav_rendezvous_m6_feedforward_ablation/2026-07-23_00-11-43_m6_ff_ablation_300_seed42/model_299.pt`.
- M6 recurrent training and hidden-state management are functional, but a measurable implicit-prediction advantage over the fair feedforward baseline was not demonstrated.

## Task

```text
Isaac-Uav-Rendezvous-Direct-v0
Isaac-Uav-Rendezvous-Baseline-v0
Isaac-Uav-Rendezvous-RL-v0
Isaac-Uav-Rendezvous-Recurrent-v0
Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0
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

Run M5 RL runtime audit:

```bash
env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u VIRTUAL_ENV \
  /home/lab_726/IsaacLab/isaaclab.sh -p \
  /home/lab_726/uav_rendezvous_rl/scripts/audit_m5_rl_runtime.py \
  --scenario all \
  --num_envs 64 \
  --steps 10000 \
  --seed 42 \
  --device cuda:0 \
  --headless
```

Run M5 short PPO sanity training:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/train.py --task Isaac-Uav-Rendezvous-RL-v0 --num_envs 64 --max_iterations 5 --seed 42 --device cuda:0 --headless --run_name m5_startup
```

Run final M5 deterministic validation:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/evaluate.py --task Isaac-Uav-Rendezvous-RL-v0 --policy trained --num_envs 64 --episodes 4 --seed 4242 --split validation --checkpoint /home/lab_726/uav_rendezvous_rl/logs/rsl_rl/uav_rendezvous_m5_rl/2026-07-22_19-04-26_m5_rewardfix_300_seed42/model_299.pt --device cuda:0 --headless
```

Run M6 recurrent runtime audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m6_recurrent_runtime.py --task Isaac-Uav-Rendezvous-Recurrent-v0 --num_envs 4 --steps 1 --seed 42 --device cuda:0 --headless
```

Run M6 history-sensitivity audit:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/audit_m6_history_sensitivity.py --task Isaac-Uav-Rendezvous-Recurrent-v0 --num_envs 4 --history_steps 8 --seed 42 --device cuda:0 --headless
```

Run M6 recurrent startup training:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV /home/lab_726/IsaacLab/isaaclab.sh -p /home/lab_726/uav_rendezvous_rl/scripts/train.py --task Isaac-Uav-Rendezvous-Recurrent-v0 --num_envs 4 --max_iterations 1 --seed 42 --device cuda:0 --headless --run_name m6_startup
```
