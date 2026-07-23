#!/usr/bin/env python3
"""Evaluate PPO checkpoints deterministically."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
import math
import os
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic checkpoint evaluation for UAV rendezvous RL tasks.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per environment to evaluate.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-RL-v0", help="Gymnasium task ID.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation", help="Target-motion split.")
    parser.add_argument(
        "--target_motion_mode",
        choices=("Mixed", "ConstantVelocity", "ConstantAcceleration", "ConstantTurn", "PiecewiseAcceleration"),
        default=None,
        help="Override target-motion mode distribution for validation.",
    )
    parser.add_argument(
        "--force_mode_cycle_on_reset",
        action="store_true",
        default=False,
        help="Cycle modes by env id on reset for balanced per-mode episode counts.",
    )
    parser.add_argument(
        "--policy",
        choices=("trained", "zero", "random", "oracle"),
        default="trained",
        help="Policy/action source to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for environment and PPO runner.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to latest run checkpoint.")
    parser.add_argument("--load_run", type=str, default=".*", help="Run directory regex when checkpoint is omitted.")
    parser.add_argument("--load_checkpoint", type=str, default="model_.*.pt", help="Checkpoint regex when omitted.")
    parser.add_argument("--oracle_gain", type=float, default=0.8, help="Current-state proportional oracle gain.")
    parser.add_argument(
        "--determinism_check",
        action="store_true",
        default=False,
        help="Verify deterministic inference from the same reset hidden state and observation.",
    )
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
from uav_rendezvous_rl.controllers import clamp_vector_norm  # noqa: E402
from uav_rendezvous_rl.mdp import raw_action_from_velocity_command  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _assert_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite tensor detected in {name}.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(f"{name}.{key}", item)


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


def _configure_target_motion(env_cfg: object, mode_name: str | None, force_cycle: bool) -> None:
    if mode_name is None and not force_cycle:
        return
    probabilities = _mode_probabilities(mode_name or "Mixed")
    motion_cfg = env_cfg.target_motion
    env_cfg.target_motion = replace(
        motion_cfg,
        force_mode_cycle_on_reset=force_cycle,
        train=replace(motion_cfg.train, mode_probabilities=probabilities),
        validation=replace(motion_cfg.validation, mode_probabilities=probabilities),
        test=replace(motion_cfg.test, mode_probabilities=probabilities),
    )


def _resolve_checkpoint(agent_cfg: Any) -> str:
    if args_cli.checkpoint is not None:
        checkpoint = os.path.abspath(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.load_checkpoint)
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def _policy_device(policy_nn: Any) -> torch.device:
    return next(policy_nn.parameters()).device


def _obs_to_tensordict(policy_nn: Any, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
    device = _policy_device(policy_nn)
    batch_size = [int(obs_dict["policy"].shape[0])]
    return TensorDict({key: value.to(device=device) for key, value in obs_dict.items()}, batch_size=batch_size)


def _reset_policy(policy_nn: Any, dones: torch.Tensor | None = None) -> None:
    with torch.inference_mode():
        if dones is None:
            policy_nn.reset()
        else:
            policy_nn.reset(dones.to(device=_policy_device(policy_nn)))


def _deterministic_actions(policy_nn: Any, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    action_device = obs_dict["policy"].device
    actions = policy_nn.act_inference(_obs_to_tensordict(policy_nn, obs_dict))
    return actions.to(device=action_device)


def _clone_obs_dict(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in obs_dict.items()}


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
    collision_episodes = sum(1 for episode in selected if int(episode["collision_risk_count"]) > 0)
    workspace_episodes = sum(1 for episode in selected if int(episode["workspace_violation_count"]) > 0)
    height_episodes = sum(1 for episode in selected if int(episode["height_violation_count"]) > 0)
    speed_episodes = sum(1 for episode in selected if int(episode["speed_limit_count"]) > 0)
    return {
        "completed_episodes": count,
        "expected_episodes": expected_count,
        "success_count": len(successes),
        "success_rate": len(successes) / float(max(count, 1)),
        "collision_risk_count": sum(int(episode["collision_risk_count"]) for episode in selected),
        "collision_risk_rate": collision_episodes / float(max(count, 1)),
        "workspace_violation_count": sum(int(episode["workspace_violation_count"]) for episode in selected),
        "workspace_violation_rate": workspace_episodes / float(max(count, 1)),
        "height_violation_count": sum(int(episode["height_violation_count"]) for episode in selected),
        "height_violation_rate": height_episodes / float(max(count, 1)),
        "speed_limit_count": sum(int(episode["speed_limit_count"]) for episode in selected),
        "speed_violation_rate": speed_episodes / float(max(count, 1)),
        "average_return": float(torch.mean(torch.tensor([float(episode["episode_reward_sum"]) for episode in selected])).item())
        if selected
        else math.nan,
        "final_offset_error": _stats([float(episode["final_offset_error"]) for episode in selected]),
        "final_relative_speed": _stats([float(episode["final_relative_speed"]) for episode in selected]),
        "success_offset_error": _finite_stats([float(episode["success_offset_error"]) for episode in successes]),
        "success_relative_speed": _finite_stats([float(episode["success_relative_speed"]) for episode in successes]),
        "convergence_time": _finite_stats([float(episode["convergence_time"]) for episode in successes]),
        "minimum_center_distance": _stats([float(episode["minimum_center_distance"]) for episode in selected]),
        "episode_length_s": _stats([float(episode["episode_length_s"]) for episode in selected]),
        "action_saturation_fraction": _stats([float(episode["action_saturation_fraction"]) for episode in selected]),
        "acceleration_saturation_fraction": _stats(
            [float(episode["acceleration_saturation_fraction"]) for episode in selected]
        ),
        "episode_reward_sum": _stats([float(episode["episode_reward_sum"]) for episode in selected]),
    }


def _summarize_by_mode(history: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    selected = history[:expected_count]
    modes = sorted({str(episode.get("target_motion_mode", "Unknown")) for episode in selected})
    report = {}
    for mode in modes:
        mode_history = [episode for episode in selected if episode.get("target_motion_mode", "Unknown") == mode]
        report[mode] = _summarize(mode_history, len(mode_history))
    return report


def _target_asset_local_position(task: Any) -> torch.Tensor:
    return task.target.data.root_pos_w - task.scene.env_origins


def _ego_asset_local_position(task: Any) -> torch.Tensor:
    return task.ego.data.root_pos_w - task.scene.env_origins


def _asset_sync_errors(task: Any) -> dict[str, float]:
    return {
        "target_position": float(torch.max(torch.abs(_target_asset_local_position(task) - task.p_target_w)).item()),
        "target_velocity": float(torch.max(torch.abs(task.target.data.root_lin_vel_w - task.v_target_w)).item()),
        "ego_position": float(torch.max(torch.abs(_ego_asset_local_position(task) - task.p_ego_w)).item()),
        "ego_velocity": float(torch.max(torch.abs(task.ego.data.root_lin_vel_w - task.v_ego_w)).item()),
    }


def _max_asset_sync_errors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: max(left.get(key, 0.0), right.get(key, 0.0)) for key in set(left) | set(right)}


def _oracle_actions(task: Any) -> torch.Tensor:
    p_goal_w = task.p_target_w + task.b_des_w
    v_des_w = task.v_target_w + float(args_cli.oracle_gain) * (p_goal_w - task.p_ego_w)
    v_cmd_w, _ = clamp_vector_norm(v_des_w, task.cfg.action.v_max * 0.98)
    return raw_action_from_velocity_command(v_cmd_w, task.cfg.action.v_max)


def _actions_for_policy(
    policy_name: str,
    task: Any,
    policy_nn: Any | None,
    obs_dict: dict[str, torch.Tensor],
    random_generator: torch.Generator,
) -> torch.Tensor:
    if policy_name == "trained":
        if policy_nn is None:
            raise RuntimeError("Trained policy evaluation requires a loaded policy.")
        return _deterministic_actions(policy_nn, obs_dict)
    if policy_name == "zero":
        return torch.zeros(task.action_space.shape, dtype=torch.float32, device=task.device)
    if policy_name == "random":
        return torch.randn(task.action_space.shape, dtype=torch.float32, device=task.device, generator=random_generator)
    if policy_name == "oracle":
        return _oracle_actions(task)
    raise RuntimeError(f"Unknown policy: {policy_name}")


def _lightweight_diagnostics(task: Any) -> dict[str, Any]:
    policy_obs = task.obs_buf["policy"]
    critic_obs = task.obs_buf["critic"]
    tensors = (
        task.p_ego_w,
        task.v_ego_w,
        task.p_target_w,
        task.v_target_w,
        task.a_target_w,
        task.p_rel_w,
        task.v_rel_w,
        task.e_offset_w,
        task.raw_action,
        task.squashed_action,
        task.v_cmd_w,
        task.a_cmd_w,
        task.b_des_w,
        policy_obs,
        critic_obs,
    )
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
    diagnostics = {
        "total_steps": int(task.common_step_counter),
        "policy_obs_dim": int(policy_obs.shape[1]),
        "critic_obs_dim": int(critic_obs.shape[1]),
        "finite_check": finite,
        "collision_risk_count": task.collision_risk_count.detach().cpu().tolist(),
        "workspace_violation_count": task.workspace_violation_count.detach().cpu().tolist(),
        "height_violation_count": task.height_violation_count.detach().cpu().tolist(),
        "action_saturation_fraction": task._policy_saturation_fraction(
            task.action_saturation_count, task._all_env_ids
        ).detach().cpu().tolist(),
        "acceleration_saturation_fraction": task._saturation_fraction(
            task.acceleration_saturation_count, task._all_env_ids
        ).detach().cpu().tolist(),
    }
    if hasattr(task, "target_motion_manager"):
        diagnostics["mode_counts"] = task.target_motion_manager.mode_counts()
    return diagnostics


def _episode_history(task: Any, clear: bool) -> list[dict[str, Any]]:
    if hasattr(task, "get_m6_episode_history"):
        return task.get_m6_episode_history(clear=clear)
    return task.get_m5_episode_history(clear=clear)


def _assert_deterministic_inference(policy_nn: Any, obs_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    _reset_policy(policy_nn)
    with torch.inference_mode():
        actions_a = _deterministic_actions(policy_nn, obs_dict)
    _reset_policy(policy_nn)
    with torch.inference_mode():
        actions_b = _deterministic_actions(policy_nn, obs_dict)
    _reset_policy(policy_nn)
    max_abs_delta = float(torch.max(torch.abs(actions_a - actions_b)).item())
    if max_abs_delta > 1.0e-6:
        raise RuntimeError(f"Deterministic inference check failed: max_abs_delta={max_abs_delta}.")
    return {"max_abs_delta": max_abs_delta}


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.target_motion_split = args_cli.split
    _configure_target_motion(env_cfg, args_cli.target_motion_mode, args_cli.force_mode_cycle_on_reset)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed
    agent_cfg.device = device
    checkpoint = None
    if args_cli.policy == "trained":
        checkpoint = _resolve_checkpoint(agent_cfg)
        env_cfg.log_dir = os.path.dirname(checkpoint)

    gym_env = None
    rsl_env = None
    try:
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        policy_nn = None
        if args_cli.policy == "trained":
            rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[INFO] Loading model checkpoint from: {checkpoint}", flush=True)
            runner.load(checkpoint)
            policy_nn = runner.alg.policy
            policy_nn.to("cpu")
            policy_nn.eval()
            _reset_policy(policy_nn)
        obs_dict, _ = gym_env.reset()
        obs_dict = _clone_obs_dict(obs_dict)
        task = gym_env.unwrapped
        _episode_history(task, clear=True)
        _assert_finite("reset_obs", obs_dict)
        max_sync = _asset_sync_errors(task)
        determinism_report = None
        if args_cli.determinism_check and policy_nn is not None:
            determinism_report = _assert_deterministic_inference(policy_nn, obs_dict)
        expected = int(args_cli.num_envs) * int(args_cli.episodes)
        max_steps = int(task.max_episode_length * args_cli.episodes)
        random_generator = torch.Generator(device=task.device)
        random_generator.manual_seed(int(agent_cfg.seed) + 991)
        print(
            f"[INFO] Starting evaluation policy={args_cli.policy}, split={args_cli.split}, "
            f"steps={max_steps}.",
            flush=True,
        )
        for step in range(max_steps):
            with torch.inference_mode():
                actions = _actions_for_policy(args_cli.policy, task, policy_nn, obs_dict, random_generator)
                if step == 0:
                    print(
                        f"[INFO] First inference action shape: {tuple(actions.shape)}, "
                        f"finite={bool(torch.isfinite(actions).all().item())}, "
                        f"abs_max={float(torch.abs(actions).max().item())}",
                        flush=True,
                    )
                obs_dict, rewards, terminated, truncated, _ = gym_env.step(actions)
                max_sync = _max_asset_sync_errors(max_sync, _asset_sync_errors(task))
                dones = (terminated | truncated).to(dtype=torch.long)
                if policy_nn is not None:
                    _reset_policy(policy_nn, dones)
                obs_dict = _clone_obs_dict(obs_dict)
                if step == 0:
                    print(
                        f"[INFO] First deterministic env step completed: "
                        f"terminated={int(torch.count_nonzero(terminated).item())}, "
                        f"truncated={int(torch.count_nonzero(truncated).item())}, "
                        f"reward_mean={float(torch.mean(rewards).item())}",
                        flush=True,
                    )
            _assert_finite("obs", obs_dict)
            _assert_finite("rewards", rewards)
            if step == 0:
                print("[INFO] First deterministic post-step checks completed.", flush=True)
        history = _episode_history(task, clear=True)
        report = {
            "task": args_cli.task,
            "policy": args_cli.policy,
            "checkpoint": checkpoint,
            "seed": int(agent_cfg.seed),
            "split": args_cli.split,
            "target_motion_mode": args_cli.target_motion_mode,
            "force_mode_cycle_on_reset": bool(args_cli.force_mode_cycle_on_reset),
            "num_envs": int(args_cli.num_envs),
            "episodes": int(args_cli.episodes),
            "summary": _summarize(history, expected),
            "summary_by_mode": _summarize_by_mode(history, expected),
            "diagnostics": _lightweight_diagnostics(task),
            "asset_sync_errors": max_sync,
            "determinism_check": determinism_report,
        }
        print(f"[INFO] deterministic evaluation: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
