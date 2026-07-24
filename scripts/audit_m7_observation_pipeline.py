#!/usr/bin/env python3
"""Runtime audit for the M7A causal observation pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for M7A observation degradation.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
    parser.add_argument("--steps", type=int, default=10000, help="Finite rollout steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for environment and PPO runner.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-M7A-GRU-v0", help="Gymnasium M7A task ID.")
    parser.add_argument("--m7a_stage", type=str, default="4", help="M7A observation stage to audit.")
    parser.add_argument(
        "--target_motion_mode",
        choices=("Mixed", "ConstantVelocity", "ConstantAcceleration", "ConstantTurn", "PiecewiseAcceleration"),
        default="Mixed",
        help="Target-motion distribution for the audit.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg, make_m7a_observation_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mode_probabilities(mode_name: str) -> tuple[float, float, float, float]:
    if mode_name == "Mixed":
        return (0.25, 0.25, 0.25, 0.25)
    if mode_name == "ConstantVelocity":
        return (1.0, 0.0, 0.0, 0.0)
    if mode_name == "ConstantAcceleration":
        return (0.0, 1.0, 0.0, 0.0)
    if mode_name == "ConstantTurn":
        return (0.0, 0.0, 1.0, 0.0)
    if mode_name == "PiecewiseAcceleration":
        return (0.0, 0.0, 0.0, 1.0)
    raise RuntimeError(f"Unknown target motion mode: {mode_name}")


def _configure_target_motion(env_cfg: Any) -> None:
    probabilities = _mode_probabilities(args_cli.target_motion_mode)
    motion_cfg = env_cfg.target_motion
    env_cfg.target_motion = replace(
        motion_cfg,
        train=replace(motion_cfg.train, mode_probabilities=probabilities),
        validation=replace(motion_cfg.validation, mode_probabilities=probabilities),
        test=replace(motion_cfg.test, mode_probabilities=probabilities),
    )


def _asset_sync_errors(task: Any) -> dict[str, float]:
    target_local = task.target.data.root_pos_w - task.scene.env_origins
    ego_local = task.ego.data.root_pos_w - task.scene.env_origins
    return {
        "target_position": float(torch.max(torch.abs(target_local - task.p_target_w)).item()),
        "target_velocity": float(torch.max(torch.abs(task.target.data.root_lin_vel_w - task.v_target_w)).item()),
        "ego_position": float(torch.max(torch.abs(ego_local - task.p_ego_w)).item()),
        "ego_velocity": float(torch.max(torch.abs(task.ego.data.root_lin_vel_w - task.v_ego_w)).item()),
    }


def _merge_sync_errors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: max(left.get(key, 0.0), right.get(key, 0.0)) for key in set(left) | set(right)}


def _assert_policy_hidden_reset(runner: OnPolicyRunner, obs: Any) -> dict[str, int | str | bool]:
    policy = runner.alg.policy
    _assert(policy.__class__.__name__ == "ActorCriticRecurrent", "M7A GRU task did not construct ActorCriticRecurrent.")
    _assert(bool(getattr(policy, "is_recurrent", False)), "M7A GRU policy is not recurrent.")
    _assert(isinstance(policy.memory_a.rnn, torch.nn.GRU), "Actor memory is not GRU.")
    _assert(isinstance(policy.memory_c.rnn, torch.nn.GRU), "Critic memory is not GRU.")
    policy.reset()
    with torch.inference_mode():
        policy.act_inference(obs)
        policy.evaluate(obs)
    actor_before = policy.memory_a.hidden_state.detach().clone()
    critic_before = policy.memory_c.hidden_state.detach().clone()
    dones = torch.zeros(actor_before.shape[1], dtype=torch.long, device=actor_before.device)
    dones[0] = 1
    with torch.inference_mode():
        policy.reset(dones)
    done_mask = dones == 1
    keep_mask = ~done_mask
    _assert(bool(torch.all(policy.memory_a.hidden_state[:, done_mask, :] == 0.0).item()), "Actor hidden not reset.")
    _assert(bool(torch.all(policy.memory_c.hidden_state[:, done_mask, :] == 0.0).item()), "Critic hidden not reset.")
    if bool(torch.any(keep_mask).item()):
        actor_delta = float(torch.max(torch.abs(policy.memory_a.hidden_state[:, keep_mask, :] - actor_before[:, keep_mask, :])).item())
        critic_delta = float(torch.max(torch.abs(policy.memory_c.hidden_state[:, keep_mask, :] - critic_before[:, keep_mask, :])).item())
        _assert(actor_delta <= 1.0e-6, f"Non-done actor hidden changed: {actor_delta}.")
        _assert(critic_delta <= 1.0e-6, f"Non-done critic hidden changed: {critic_delta}.")
    policy.reset()
    return {"policy_class": policy.__class__.__name__, "is_recurrent": True, "done_count": 1, "kept_count": int(actor_before.shape[1] - 1)}


def _assert_partial_reset(task: Any) -> dict[str, Any]:
    before = task.observation_pipeline.runtime_state()
    reset_ids = torch.arange(0, min(2, task.num_envs), dtype=torch.long, device=task.device)
    if reset_ids.numel() == 0:
        return {"checked": False}
    keep_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    keep_mask[reset_ids] = False
    keep_ids = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    task._reset_idx(reset_ids)
    after = task.observation_pipeline.runtime_state()
    for key, value in before.items():
        if keep_ids.numel() > 0:
            _assert(torch.equal(after[key][keep_ids], value[keep_ids]), f"Partial reset changed unselected state: {key}.")
    _assert(bool(torch.all(after["step_count"][reset_ids] == 0).item()), "Partial reset did not clear selected counters.")
    return {"checked": True, "reset_count": int(reset_ids.numel()), "kept_count": int(keep_ids.numel())}


def _assert_no_future_leakage() -> dict[str, Any]:
    sample = lambda value: torch.tensor([[value, 0.0, 0.0]], dtype=torch.float32)
    pipeline = ObservationPipeline(ObservationPipelineCfg(position_delay_steps=2, velocity_delay_steps=1), 1, "cpu", seed=77)
    pipeline.reset(None, sample(0.0), sample(100.0))
    pulse = []
    ramp = []
    for step in range(7):
        p_value = 10.0 if step == 3 else 0.0
        p_obs, v_obs = pipeline.observe(sample(p_value), sample(float(100 + step)))
        pulse.append(float(p_obs[0, 0].item()))
        ramp.append(float(v_obs[0, 0].item()))
    _assert(pulse == [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0], f"Pulse leakage/off-by-one: {pulse}.")
    _assert(ramp == [100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0], f"Ramp off-by-one: {ramp}.")
    return {"pulse": pulse, "ramp": ramp}


def _noise_report() -> dict[str, float]:
    cfg = ObservationPipelineCfg(position_noise_std=0.05, velocity_noise_std=0.05)
    pipeline = ObservationPipeline(cfg, 8192, "cpu", seed=99)
    truth = torch.zeros((8192, 3), dtype=torch.float32)
    pipeline.reset(None, truth, truth)
    p_obs, v_obs = pipeline.observe(truth, truth)
    return {
        "position_mean": float(p_obs.mean().item()),
        "position_std": float(p_obs.std(unbiased=False).item()),
        "velocity_mean": float(v_obs.mean().item()),
        "velocity_std": float(v_obs.std(unbiased=False).item()),
    }


def _registration_guard() -> dict[str, str]:
    expected = {
        "Isaac-Uav-Rendezvous-Direct-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_env:UavRendezvousEnv",
        "Isaac-Uav-Rendezvous-Baseline-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_baseline_env:UavRendezvousBaselineEnv",
        "Isaac-Uav-Rendezvous-RL-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_rl_env:UavRendezvousRLEnv",
        "Isaac-Uav-Rendezvous-Recurrent-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
        "Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
    }
    for task_id, entry_point in expected.items():
        _assert(gym.spec(task_id).entry_point == entry_point, f"Registration changed for {task_id}.")
    return expected


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    env_cfg.observation_degradation = make_m7a_observation_cfg(args_cli.m7a_stage)
    _configure_target_motion(env_cfg)

    gym_env = None
    rsl_env = None
    try:
        _registration_guard()
        no_future_leakage = _assert_no_future_leakage()
        noise = _noise_report()
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        obs = rsl_env.get_observations().to(agent_cfg.device)
        hidden_reset = _assert_policy_hidden_reset(runner, obs)
        task = gym_env.unwrapped
        partial_reset = _assert_partial_reset(task)
        obs = rsl_env.get_observations().to(agent_cfg.device)
        max_sync = _asset_sync_errors(task)
        done_count = 0
        reward_sum = 0.0
        for _ in range(int(args_cli.steps)):
            actions = torch.zeros((task.num_envs, 3), dtype=torch.float32, device=task.device)
            with torch.inference_mode():
                obs, rewards, dones, _ = rsl_env.step(actions)
                runner.alg.policy.reset(dones.to(device=agent_cfg.device))
                obs = obs.to(agent_cfg.device)
            _assert(bool(torch.isfinite(obs["policy"]).all().item()), "Non-finite policy observation.")
            _assert(bool(torch.isfinite(obs["critic"]).all().item()), "Non-finite critic observation.")
            _assert(bool(torch.isfinite(rewards).all().item()), "Non-finite rewards.")
            max_sync = _merge_sync_errors(max_sync, _asset_sync_errors(task))
            done_count += int(torch.count_nonzero(dones).item())
            reward_sum += float(torch.sum(rewards).item())
        diagnostics = task.get_m7a_diagnostics()
        _assert(diagnostics["policy_obs_dim"] == 25, "M7A policy observation dim changed.")
        _assert(diagnostics["critic_obs_dim"] == 57, "M7A critic observation dim changed.")
        _assert(bool(diagnostics["finite_check"]), "M7A diagnostics finite check failed.")
        report = {
            "task": args_cli.task,
            "m7a_stage": args_cli.m7a_stage,
            "seed": int(args_cli.seed),
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "done_count": done_count,
            "reward_sum": reward_sum,
            "hidden_reset": hidden_reset,
            "partial_reset": partial_reset,
            "no_future_leakage": no_future_leakage,
            "noise_report": noise,
            "diagnostics": diagnostics,
            "asset_sync_errors": max_sync,
        }
        print(f"[INFO] M7A observation pipeline audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
