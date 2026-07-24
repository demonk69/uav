#!/usr/bin/env python3
"""Compare trained M7A GRU and feedforward policies under one observation stage."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M7A GRU/feedforward POMDP comparison.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--episodes", type=int, default=8, help="Episodes per environment.")
    parser.add_argument("--seed", type=int, default=4242, help="Validation seed.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--m7a_stage", type=str, default="0", help="M7A observation stage.")
    parser.add_argument("--gru_checkpoint", type=str, required=True, help="Trained GRU checkpoint.")
    parser.add_argument("--feedforward_checkpoint", type=str, required=True, help="Trained feedforward checkpoint.")
    parser.add_argument("--target_motion_mode", choices=("Mixed", "ConstantVelocity", "ConstantAcceleration", "ConstantTurn", "PiecewiseAcceleration"), default="Mixed")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from uav_rendezvous_rl.observations import make_m7a_observation_cfg  # noqa: E402
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


def _configure_env(env_cfg: Any) -> None:
    env_cfg.seed = int(args_cli.seed)
    env_cfg.target_motion_split = args_cli.split
    env_cfg.observation_degradation = make_m7a_observation_cfg(args_cli.m7a_stage)
    probabilities = _mode_probabilities(args_cli.target_motion_mode)
    motion_cfg = env_cfg.target_motion
    env_cfg.target_motion = replace(
        motion_cfg,
        train=replace(motion_cfg.train, mode_probabilities=probabilities),
        validation=replace(motion_cfg.validation, mode_probabilities=probabilities),
        test=replace(motion_cfg.test, mode_probabilities=probabilities),
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan, "max": math.nan}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(torch.mean(tensor).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(torch.max(tensor).item()),
    }


def _finite_stats(values: list[float]) -> dict[str, float]:
    return _stats([value for value in values if math.isfinite(value)])


def _summarize(history: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    selected = history[:expected_count]
    successes = [episode for episode in selected if bool(episode["success"])]
    count = len(selected)
    return {
        "completed_episodes": count,
        "expected_episodes": expected_count,
        "success_count": len(successes),
        "success_rate": len(successes) / float(max(count, 1)),
        "collision_risk_count": sum(int(episode["collision_risk_count"]) for episode in selected),
        "collision_risk_rate": sum(1 for episode in selected if int(episode["collision_risk_count"]) > 0) / float(max(count, 1)),
        "workspace_violation_count": sum(int(episode["workspace_violation_count"]) for episode in selected),
        "height_violation_count": sum(int(episode["height_violation_count"]) for episode in selected),
        "speed_limit_count": sum(int(episode["speed_limit_count"]) for episode in selected),
        "average_return": float(torch.mean(torch.tensor([float(episode["episode_reward_sum"]) for episode in selected])).item()) if selected else math.nan,
        "success_offset_error": _finite_stats([float(episode["success_offset_error"]) for episode in successes]),
        "success_relative_speed": _finite_stats([float(episode["success_relative_speed"]) for episode in successes]),
        "convergence_time": _finite_stats([float(episode["convergence_time"]) for episode in successes]),
        "action_saturation_fraction": _stats([float(episode["action_saturation_fraction"]) for episode in selected]),
        "acceleration_saturation_fraction": _stats([float(episode["acceleration_saturation_fraction"]) for episode in selected]),
    }


def _summarize_by_mode(history: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    selected = history[:expected_count]
    modes = sorted({str(episode.get("target_motion_mode", "Unknown")) for episode in selected})
    return {mode: _summarize([episode for episode in selected if episode.get("target_motion_mode") == mode], len([episode for episode in selected if episode.get("target_motion_mode") == mode])) for mode in modes}


def _obs_to_tensordict(policy: Any, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
    device = next(policy.parameters()).device
    batch_size = [int(obs_dict["policy"].shape[0])]
    return TensorDict({key: value.to(device=device) for key, value in obs_dict.items()}, batch_size=batch_size)


def _reset_policy(policy: Any, dones: torch.Tensor | None = None) -> None:
    with torch.inference_mode():
        if dones is None:
            policy.reset()
        else:
            policy.reset(dones.to(device=next(policy.parameters()).device))


def _create_eval_env() -> tuple[Any, Any]:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    task_id = "Isaac-Uav-Rendezvous-M7A-GRU-v0"
    print(f"[INFO] Creating shared M7A comparison env from {task_id}.", flush=True)
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    _configure_env(env_cfg)
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.device = device
    gym_env = gym.make(task_id, cfg=env_cfg)
    rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    return gym_env, rsl_env


def _evaluate(task_id: str, checkpoint: str, gym_env: Any, rsl_env: Any) -> dict[str, Any]:
    print(f"[INFO] Starting M7A comparison evaluation for {task_id}.", flush=True)
    _assert(os.path.isfile(checkpoint), f"Checkpoint does not exist: {checkpoint}")
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.device = device
    print(f"[INFO] Creating runner for {task_id}.", flush=True)
    runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint for {task_id}: {checkpoint}", flush=True)
    runner.load(os.path.abspath(checkpoint))
    policy = runner.alg.policy
    policy.eval()
    _reset_policy(policy)
    print(f"[INFO] Resetting shared env for {task_id}.", flush=True)
    obs_dict, _ = gym_env.reset()
    obs_dict = {key: value.detach().clone() for key, value in obs_dict.items()}
    task = gym_env.unwrapped
    task.get_m7a_episode_history(clear=True)
    expected = int(args_cli.num_envs) * int(args_cli.episodes)
    max_steps = int(task.max_episode_length * args_cli.episodes)
    print(f"[INFO] Running {task_id}: max_steps={max_steps}, expected_episodes={expected}.", flush=True)
    for _ in range(max_steps):
        with torch.inference_mode():
            actions = policy.act_inference(_obs_to_tensordict(policy, obs_dict)).to(device=task.device)
            obs_dict, rewards, terminated, truncated, _ = gym_env.step(actions)
            _assert(bool(torch.isfinite(obs_dict["policy"]).all().item()), "Non-finite policy observation.")
            _assert(bool(torch.isfinite(obs_dict["critic"]).all().item()), "Non-finite critic observation.")
            _assert(bool(torch.isfinite(rewards).all().item()), "Non-finite reward.")
            _reset_policy(policy, (terminated | truncated).to(dtype=torch.long))
            obs_dict = {key: value.detach().clone() for key, value in obs_dict.items()}
            if len(task.get_m7a_episode_history(clear=False)) >= expected:
                break
    history = task.get_m7a_episode_history(clear=True)
    diagnostics = task.get_m7a_diagnostics()
    print(f"[INFO] Completed {task_id}: episodes={len(history)}.", flush=True)
    return {
        "task": task_id,
        "checkpoint": os.path.abspath(checkpoint),
        "summary": _summarize(history, expected),
        "summary_by_mode": _summarize_by_mode(history, expected),
        "diagnostics": diagnostics,
    }


def _advantage(gru: dict[str, Any], ff: dict[str, Any]) -> dict[str, Any]:
    gru_summary = gru["summary"]
    ff_summary = ff["summary"]
    success_delta = float(gru_summary["success_rate"] - ff_summary["success_rate"])
    offset_ff = float(ff_summary["success_offset_error"]["p95"])
    speed_ff = float(ff_summary["success_relative_speed"]["p95"])
    convergence_ff = float(ff_summary["convergence_time"]["mean"])
    offset_reduction = (offset_ff - float(gru_summary["success_offset_error"]["p95"])) / max(offset_ff, 1.0e-6)
    speed_reduction = (speed_ff - float(gru_summary["success_relative_speed"]["p95"])) / max(speed_ff, 1.0e-6)
    convergence_reduction = (convergence_ff - float(gru_summary["convergence_time"]["mean"])) / max(convergence_ff, 1.0e-6)
    safe = float(gru_summary["collision_risk_rate"]) <= float(ff_summary["collision_risk_rate"])
    return {
        "success_delta": success_delta,
        "offset_p95_reduction_fraction": offset_reduction,
        "relative_speed_p95_reduction_fraction": speed_reduction,
        "convergence_time_reduction_fraction": convergence_reduction,
        "return_delta": float(gru_summary["average_return"] - ff_summary["average_return"]),
        "safety_not_worse": bool(safe),
        "history_advantage_claim_allowed": bool(
            safe
            and (
                success_delta >= 0.05
                or offset_reduction >= 0.10
                or speed_reduction >= 0.10
                or convergence_reduction >= 0.10
                or float(gru_summary["average_return"]) > float(ff_summary["average_return"])
            )
        ),
    }


def main() -> None:
    gym_env = None
    rsl_env = None
    try:
        gym_env, rsl_env = _create_eval_env()
        gru = _evaluate("Isaac-Uav-Rendezvous-M7A-GRU-v0", args_cli.gru_checkpoint, gym_env, rsl_env)
        feedforward = _evaluate(
            "Isaac-Uav-Rendezvous-M7A-Feedforward-v0", args_cli.feedforward_checkpoint, gym_env, rsl_env
        )
        report = {
            "m7a_stage": args_cli.m7a_stage,
            "seed": int(args_cli.seed),
            "split": args_cli.split,
            "target_motion_mode": args_cli.target_motion_mode,
            "num_envs": int(args_cli.num_envs),
            "episodes": int(args_cli.episodes),
            "gru": gru,
            "feedforward": feedforward,
            "advantage": _advantage(gru, feedforward),
        }
        print(f"[INFO] M7A POMDP comparison: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
