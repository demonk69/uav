#!/usr/bin/env python3
"""Runtime audit for the M5 feedforward RL environment under Isaac Lab."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for the M5 UAV rendezvous RL task.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--steps", type=int, default=10000, help="Environment steps per scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for scenarios.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train", help="Motion split to audit.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-RL-v0", help="Gymnasium M5 task ID.")
    parser.add_argument(
        "--scenario",
        choices=("all", "zero", "random", "mixed", "oracle"),
        default="all",
        help="Scenario to run.",
    )
    parser.add_argument("--oracle_gain", type=float, default=0.8, help="Current-state proportional oracle gain.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from uav_rendezvous_rl.controllers import clamp_vector_norm  # noqa: E402
from uav_rendezvous_rl.mdp import raw_action_from_velocity_command  # noqa: E402
from uav_rendezvous_rl.motions.sampling import make_split_generator  # noqa: E402


SYNC_POSITION_TOL = 1.0e-4
SYNC_VELOCITY_TOL = 1.0e-4


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_all_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _assert(bool(torch.isfinite(value).all().item()), f"Non-finite tensor detected in {name}.")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_all_finite(f"{name}.{key}", item)


def _mode_probabilities(mode_name: str) -> tuple[float, float, float, float]:
    if mode_name == "ConstantVelocity":
        return (1.0, 0.0, 0.0, 0.0)
    if mode_name == "Mixed":
        return (0.25, 0.25, 0.25, 0.25)
    raise ValueError(f"Unknown mode name: {mode_name}")


def _configured_motion_cfg(base_motion_cfg: Any, mode_name: str) -> Any:
    probabilities = _mode_probabilities(mode_name)
    train = replace(base_motion_cfg.train, mode_probabilities=probabilities)
    validation = replace(base_motion_cfg.validation, mode_probabilities=probabilities)
    test = replace(base_motion_cfg.test, mode_probabilities=probabilities)
    return replace(
        base_motion_cfg,
        force_mode_cycle_on_reset=mode_name == "Mixed",
        train=train,
        validation=validation,
        test=test,
    )


def _make_env() -> Any:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.target_motion_split = args_cli.split
    return gym.make(args_cli.task, cfg=env_cfg)


def _configure_task_for_scenario(task: Any, scenario_seed: int, mode_name: str, base_motion_cfg: Any) -> None:
    motion_cfg = _configured_motion_cfg(base_motion_cfg, mode_name)
    task.cfg.seed = scenario_seed
    task.cfg.target_motion_split = args_cli.split
    task.cfg.target_motion = motion_cfg
    task.target_motion_manager.cfg = motion_cfg
    task._base_seed_value = scenario_seed
    task._seed_value = scenario_seed
    task._rng = make_split_generator(scenario_seed, motion_cfg, args_cli.split)


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


def _assert_observation_contract(task: Any, obs: dict[str, torch.Tensor]) -> None:
    policy = obs["policy"]
    critic = obs["critic"]
    _assert(policy.shape == (task.num_envs, 25), f"Expected policy obs shape {(task.num_envs, 25)}, got {policy.shape}.")
    _assert(critic.shape == (task.num_envs, 57), f"Expected critic obs shape {(task.num_envs, 57)}, got {critic.shape}.")
    _assert(bool(torch.allclose(policy[:, 0:3], task.p_rel_w, atol=1.0e-6)), "Policy p_rel_w slice mismatch.")
    _assert(bool(torch.allclose(policy[:, 3:6], task.v_rel_w, atol=1.0e-6)), "Policy v_rel_w slice mismatch.")
    _assert(bool(torch.allclose(policy[:, 6:9], task.v_ego_w, atol=1.0e-6)), "Policy v_ego_w slice mismatch.")
    _assert(
        bool(torch.allclose(policy[:, 18:21], task.previous_squashed_action, atol=1.0e-6)),
        "Policy last_action slice is not previous_squashed_action.",
    )
    _assert(bool(torch.allclose(policy[:, 21:24], task.b_des_w, atol=1.0e-6)), "Policy b_des_w slice mismatch.")
    _assert(bool(torch.allclose(critic[:, 0:25], policy, atol=1.0e-6)), "Critic actor-prefix mismatch.")


def _assert_action_contract(task: Any) -> None:
    expected_squashed = torch.tanh(task.raw_action)
    expected_v_cmd = expected_squashed * float(task.cfg.action.v_max)
    _assert(bool(torch.allclose(task.squashed_action, expected_squashed, atol=1.0e-6)), "Squashed action mismatch.")
    _assert(bool(torch.allclose(task.v_cmd_w, expected_v_cmd, atol=1.0e-6)), "Velocity command mapping mismatch.")
    _assert(float(torch.abs(task.squashed_action).max().item()) <= 1.0 + 1.0e-6, "Squashed action outside [-1, 1].")
    _assert(
        float(torch.linalg.norm(task.v_cmd_w, dim=1).max().item()) <= (3.0**0.5) * task.cfg.action.v_max + 1.0e-6,
        "Velocity command component mapping exceeded expected tanh bounds.",
    )


def _oracle_action(task: Any) -> torch.Tensor:
    p_goal_w = task.p_target_w + task.b_des_w
    v_des_w = task.v_target_w + float(args_cli.oracle_gain) * (p_goal_w - task.p_ego_w)
    v_cmd_w, _ = clamp_vector_norm(v_des_w, task.cfg.action.v_max * 0.98)
    return raw_action_from_velocity_command(v_cmd_w, task.cfg.action.v_max)


def _actions_for_scenario(task: Any, action_mode: str) -> torch.Tensor:
    if action_mode == "zero":
        return torch.zeros(task.action_space.shape, dtype=torch.float32, device=task.device)
    if action_mode == "random":
        return torch.randn(task.action_space.shape, dtype=torch.float32, device=task.device)
    if action_mode == "oracle":
        return _oracle_action(task)
    raise ValueError(f"Unknown action mode: {action_mode}")


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


def _summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [episode for episode in history if bool(episode["success"])]
    count = len(history)
    return {
        "completed_episodes": count,
        "success_count": len(successes),
        "success_rate": len(successes) / float(max(count, 1)),
        "collision_risk_count": sum(int(episode["collision_risk_count"]) for episode in history),
        "workspace_violation_count": sum(int(episode["workspace_violation_count"]) for episode in history),
        "height_violation_count": sum(int(episode["height_violation_count"]) for episode in history),
        "speed_limit_count": sum(int(episode["speed_limit_count"]) for episode in history),
        "final_offset_error": _stats([float(episode["final_offset_error"]) for episode in history]),
        "final_relative_speed": _stats([float(episode["final_relative_speed"]) for episode in history]),
        "minimum_center_distance": _stats([float(episode["minimum_center_distance"]) for episode in history]),
        "episode_reward_sum": _stats([float(episode["episode_reward_sum"]) for episode in history]),
    }


def _run_scenario(env: Any, scenario_name: str, action_mode: str, mode_name: str, base_motion_cfg: Any) -> dict[str, Any]:
    scenario_seed = int(args_cli.seed) + {
        "zero_constant_velocity": 0,
        "random_constant_velocity": 1000,
        "random_mixed_modes": 2000,
        "oracle_constant_velocity": 3000,
    }[scenario_name]
    task = env.unwrapped
    _configure_task_for_scenario(task, scenario_seed, mode_name, base_motion_cfg)
    obs, _ = env.reset(seed=scenario_seed)
    task.get_m5_episode_history(clear=True)
    _assert_all_finite("reset_obs", obs)
    _assert_observation_contract(task, obs)
    max_sync = _asset_sync_errors(task)

    for _ in range(args_cli.steps):
        actions = _actions_for_scenario(task, action_mode)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        _assert_all_finite("obs", obs)
        _assert_all_finite("rewards", rewards)
        _assert(terminated.shape == truncated.shape, "Terminated and truncated tensors have mismatched shapes.")
        _assert_observation_contract(task, obs)
        _assert_action_contract(task)
        max_sync = _max_asset_sync_errors(max_sync, _asset_sync_errors(task))

    history = task.get_m5_episode_history(clear=True)
    diagnostics = task.get_m5_diagnostics()
    _assert(bool(diagnostics["finite_check"]), f"{scenario_name} diagnostics reported non-finite state.")
    _assert(max_sync["target_position"] <= SYNC_POSITION_TOL, "Target asset position sync error too large.")
    _assert(max_sync["target_velocity"] <= SYNC_VELOCITY_TOL, "Target asset velocity sync error too large.")
    _assert(max_sync["ego_position"] <= SYNC_POSITION_TOL, "Ego asset position sync error too large.")
    _assert(max_sync["ego_velocity"] <= SYNC_VELOCITY_TOL, "Ego asset velocity sync error too large.")

    summary = _summarize_history(history)
    if action_mode == "oracle":
        _assert(summary["collision_risk_count"] == 0, "Oracle scenario reported collision risk.")
        _assert(summary["success_rate"] >= 0.80, "Oracle scenario success rate below 80%.")

    return {
        "scenario": scenario_name,
        "action_mode": action_mode,
        "motion_mode": mode_name,
        "seed": scenario_seed,
        "num_envs": int(args_cli.num_envs),
        "steps": int(args_cli.steps),
        "summary": summary,
        "asset_sync_errors": max_sync,
        "diagnostics": diagnostics,
    }


def main() -> None:
    env = None
    try:
        env = _make_env()
        task = env.unwrapped
        base_motion_cfg = task.cfg.target_motion
        scenario_specs = {
            "zero": (("zero_constant_velocity", "zero", "ConstantVelocity"),),
            "random": (("random_constant_velocity", "random", "ConstantVelocity"),),
            "mixed": (("random_mixed_modes", "random", "Mixed"),),
            "oracle": (("oracle_constant_velocity", "oracle", "ConstantVelocity"),),
            "all": (
                ("zero_constant_velocity", "zero", "ConstantVelocity"),
                ("random_constant_velocity", "random", "ConstantVelocity"),
                ("random_mixed_modes", "random", "Mixed"),
                ("oracle_constant_velocity", "oracle", "ConstantVelocity"),
            ),
        }[args_cli.scenario]
        report = {}
        for scenario_name, action_mode, mode_name in scenario_specs:
            print(f"[INFO] Running M5 scenario {scenario_name}.", flush=True)
            report[scenario_name] = _run_scenario(env, scenario_name, action_mode, mode_name, base_motion_cfg)
        print(f"[INFO] M5 RL runtime audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
