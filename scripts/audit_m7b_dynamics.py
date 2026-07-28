#!/usr/bin/env python3
"""Runtime audit for M7B dynamics randomization, action delay, and wind."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for M7B dynamics robustness infrastructure.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
    parser.add_argument("--steps", type=int, default=10000, help="Finite rollout steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for environment and runner.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-M7B-Feedforward-v0", help="Gymnasium M7B task ID.")
    parser.add_argument("--m7b_stage", type=str, default="0", help="M7B stage to audit: 0, 1, 2, 3, or 4.")
    parser.add_argument(
        "--policy",
        choices=("zero", "random"),
        default="zero",
        help="Action source for finite rollout.",
    )
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


def _configure_m7b_stage(env_cfg: Any) -> None:
    stage_id = str(args_cli.m7b_stage).strip()
    if stage_id == "0":
        env_cfg.tau_velocity_scale = 1.0
        env_cfg.acceleration_limit_scale = 1.0
        env_cfg.speed_limit_scale = 1.0
        env_cfg.linear_drag = 0.0
        env_cfg.action_delay_steps = 0
        env_cfg.steady_wind_max_magnitude = 0.0
        env_cfg.gust_max_magnitude = 0.0
    elif stage_id == "1":
        env_cfg.tau_velocity_scale = 1.50
        env_cfg.acceleration_limit_scale = 1.20
        env_cfg.speed_limit_scale = 1.10
        env_cfg.linear_drag = 0.15
        env_cfg.action_delay_steps = 0
        env_cfg.steady_wind_max_magnitude = 0.0
        env_cfg.gust_max_magnitude = 0.0
    elif stage_id == "2":
        env_cfg.tau_velocity_scale = 1.0
        env_cfg.acceleration_limit_scale = 1.0
        env_cfg.speed_limit_scale = 1.0
        env_cfg.linear_drag = 0.0
        env_cfg.action_delay_steps = 3
        env_cfg.steady_wind_max_magnitude = 0.0
        env_cfg.gust_max_magnitude = 0.0
    elif stage_id == "3":
        env_cfg.tau_velocity_scale = 1.0
        env_cfg.acceleration_limit_scale = 1.0
        env_cfg.speed_limit_scale = 1.0
        env_cfg.linear_drag = 0.0
        env_cfg.action_delay_steps = 0
        env_cfg.steady_wind_max_magnitude = 0.20
        env_cfg.gust_max_magnitude = 0.15
    elif stage_id == "4":
        env_cfg.tau_velocity_scale = 1.50
        env_cfg.acceleration_limit_scale = 1.20
        env_cfg.speed_limit_scale = 1.10
        env_cfg.linear_drag = 0.15
        env_cfg.action_delay_steps = 3
        env_cfg.steady_wind_max_magnitude = 0.20
        env_cfg.gust_max_magnitude = 0.15
    else:
        raise RuntimeError(f"Unknown M7B stage: {args_cli.m7b_stage}. Expected 0, 1, 2, 3, or 4.")


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


def _partial_reset_check(task: Any) -> dict[str, int | bool]:
    reset_ids = torch.arange(0, min(2, task.num_envs), dtype=torch.long, device=task.device)
    if reset_ids.numel() == 0:
        return {"checked": False, "reset_count": 0, "kept_count": 0}
    keep_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    keep_mask[reset_ids] = False
    keep_ids = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    fifo_before = task.action_delay_buffer.buffer.detach().clone()
    tau_before = task._m7b_tau_velocity_scale.detach().clone()
    wind_before = task._wind_state.steady_wind_w.detach().clone()

    task._reset_idx(reset_ids)

    if keep_ids.numel() > 0:
        _assert(torch.equal(task.action_delay_buffer.buffer[keep_ids], fifo_before[keep_ids]), "Partial reset changed unselected FIFO rows.")
        _assert(torch.equal(task._m7b_tau_velocity_scale[keep_ids], tau_before[keep_ids]), "Partial reset changed unselected tau scales.")
        _assert(torch.equal(task._wind_state.steady_wind_w[keep_ids], wind_before[keep_ids]), "Partial reset changed unselected wind.")
    _assert(torch.all(task.action_delay_buffer.buffer[reset_ids] == 0.0).item(), "Partial reset did not clear selected FIFO rows.")
    return {"checked": True, "reset_count": int(reset_ids.numel()), "kept_count": int(keep_ids.numel())}


def _actions(task: Any, generator: torch.Generator) -> torch.Tensor:
    if args_cli.policy == "zero":
        return torch.zeros(task.action_space.shape, dtype=torch.float32, device=task.device)
    return torch.randn(task.action_space.shape, dtype=torch.float32, device=task.device, generator=generator)


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.device = device
    _configure_target_motion(env_cfg)
    _configure_m7b_stage(env_cfg)

    gym_env = None
    rsl_env = None
    try:
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        task = gym_env.unwrapped
        obs = rsl_env.get_observations().to(agent_cfg.device)
        _assert(int(obs["policy"].shape[1]) == 25, "M7B Actor observation dimension is not 25.")
        _assert(int(obs["critic"].shape[1]) == 65, "M7B Critic observation dimension is not 65.")
        _assert(int(task.single_action_space.shape[0]) == 3, "M7B action dimension is not 3.")
        partial_reset = _partial_reset_check(task)
        obs = rsl_env.get_observations().to(agent_cfg.device)
        max_sync = _asset_sync_errors(task)
        done_count = 0
        reward_sum = 0.0
        generator = torch.Generator(device=task.device).manual_seed(int(args_cli.seed) + 1701)
        for _ in range(int(args_cli.steps)):
            with torch.inference_mode():
                obs, rewards, dones, _ = rsl_env.step(_actions(task, generator))
                obs = obs.to(agent_cfg.device)
            _assert(torch.isfinite(obs["policy"]).all().item(), "Non-finite policy observation.")
            _assert(torch.isfinite(obs["critic"]).all().item(), "Non-finite critic observation.")
            _assert(torch.isfinite(rewards).all().item(), "Non-finite rewards.")
            max_sync = _merge_sync_errors(max_sync, _asset_sync_errors(task))
            done_count += int(torch.count_nonzero(dones).item())
            reward_sum += float(torch.sum(rewards).item())
        diagnostics = task.get_m7b_diagnostics()
        _assert(bool(diagnostics["finite_check"]), "M7B diagnostics finite check failed.")
        if str(args_cli.m7b_stage).strip() == "0":
            _assert(diagnostics["tau_velocity_scale_min"] == 1.0, "M7B-S0 tau scale is not nominal.")
            _assert(diagnostics["acceleration_limit_scale_min"] == 1.0, "M7B-S0 acceleration scale is not nominal.")
            _assert(diagnostics["speed_limit_scale_min"] == 1.0, "M7B-S0 speed scale is not nominal.")
            _assert(diagnostics["linear_drag_max"] == 0.0, "M7B-S0 drag is not zero.")
            _assert(diagnostics["action_delay_steps_max"] == 0, "M7B-S0 delay is not zero.")
            _assert(diagnostics["current_wind_norm_max"] == 0.0, "M7B-S0 wind is not zero.")
        report = {
            "task": args_cli.task,
            "m7b_stage": args_cli.m7b_stage,
            "policy": args_cli.policy,
            "policy_class": runner.alg.policy.__class__.__name__,
            "seed": int(args_cli.seed),
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "done_count": done_count,
            "reward_sum": reward_sum,
            "partial_reset": partial_reset,
            "diagnostics": diagnostics,
            "asset_sync_errors": max_sync,
        }
        print(f"[INFO] M7B dynamics audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
