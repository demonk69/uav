#!/usr/bin/env python3
"""End-to-end M4 deterministic baseline runtime audit under Isaac Lab."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from typing import Any

import torch
from isaaclab.app import AppLauncher


SYNC_POSITION_TOL = 1.0e-4
SYNC_VELOCITY_TOL = 1.0e-4
TARGET_CV_TOL = 2.0e-3

gym = None
parse_env_cfg = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for the UAV rendezvous M4 deterministic baseline.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for all scenarios.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train", help="Motion split to audit.")
    parser.add_argument(
        "--task", type=str, default="Isaac-Uav-Rendezvous-Baseline-v0", help="Gymnasium M4 baseline task ID."
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _launch_app(args_cli: argparse.Namespace):
    app_launcher = AppLauncher(args_cli)
    return app_launcher.app


def _load_runtime_modules() -> None:
    global gym, parse_env_cfg
    if gym is not None and parse_env_cfg is not None:
        return

    import gymnasium as gym_module
    import uav_rendezvous_rl.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg as parse_env_cfg_fn

    gym = gym_module
    parse_env_cfg = parse_env_cfg_fn


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
    if mode_name == "ConstantAcceleration":
        return (0.0, 1.0, 0.0, 0.0)
    if mode_name == "ConstantTurn":
        return (0.0, 0.0, 1.0, 0.0)
    if mode_name == "PiecewiseAcceleration":
        return (0.0, 0.0, 0.0, 1.0)
    raise ValueError(f"Unknown mode name: {mode_name}")


def _configured_motion_cfg(base_motion_cfg: Any, mode_name: str, fixed_nominal: bool) -> Any:
    probabilities = _mode_probabilities(mode_name)
    motion_cfg = base_motion_cfg
    train = replace(motion_cfg.train, mode_probabilities=probabilities)
    validation = replace(motion_cfg.validation, mode_probabilities=probabilities)
    test = replace(motion_cfg.test, mode_probabilities=probabilities)
    if fixed_nominal:
        train = replace(
            train,
            target_pos_x_range=(4.0, 4.0),
            target_pos_y_range=(0.0, 0.0),
            target_height_range=(1.5, 1.5),
            target_vel_x_range=(0.5, 0.5),
            target_vel_y_range=(0.0, 0.0),
        )
    return replace(
        motion_cfg,
        force_mode_cycle_on_reset=False,
        train=train,
        validation=validation,
        test=test,
    )


def _configured_initial_geometry_cfg(base_initial_geometry_cfg: Any, fixed_nominal: bool) -> Any:
    if fixed_nominal:
        return replace(
            base_initial_geometry_cfg,
            delta_x_range=(3.0, 3.0),
            delta_y_range=(0.0, 0.0),
            delta_z=0.0,
        )
    return base_initial_geometry_cfg


def _make_env(args_cli: argparse.Namespace):
    _load_runtime_modules()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.target_motion_split = args_cli.split
    return gym.make(args_cli.task, cfg=env_cfg)


def _configure_task_for_scenario(
    task: Any,
    args_cli: argparse.Namespace,
    scenario_seed: int,
    mode_name: str,
    fixed_nominal: bool,
    base_motion_cfg: Any,
    base_initial_geometry_cfg: Any,
) -> None:
    from uav_rendezvous_rl.motions.sampling import make_split_generator

    motion_cfg = _configured_motion_cfg(base_motion_cfg, mode_name, fixed_nominal)
    initial_geometry_cfg = _configured_initial_geometry_cfg(base_initial_geometry_cfg, fixed_nominal)
    task.cfg.seed = scenario_seed
    task.cfg.target_motion_split = args_cli.split
    task.cfg.target_motion = motion_cfg
    task.target_motion_manager.cfg = motion_cfg
    task.cfg.baseline_initial_geometry = initial_geometry_cfg
    task._initial_geometry_cfg = initial_geometry_cfg
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


def _target_cv_error(task: Any) -> float:
    elapsed = task.target_motion_manager.motion_step_count.to(dtype=torch.float32).unsqueeze(-1) * task.physics_dt
    expected = task.target_motion_manager.p0_w + task.target_motion_manager.v0_w * elapsed
    expected[:, 2] = task.target_motion_manager.p0_w[:, 2]
    return float(torch.max(torch.abs(task.p_target_w - expected)).item())


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


def _finite_history_values(history: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for episode in history:
        value = float(episode[key])
        if math.isfinite(value):
            values.append(value)
    return values


def _summarize_history(history: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    _assert(len(history) >= expected_count, f"Expected at least {expected_count} completed episodes, got {len(history)}.")
    selected = history[:expected_count]
    successes = [episode for episode in selected if bool(episode["success"])]
    collision_count = sum(int(episode["collision_risk_count"]) for episode in selected)
    workspace_count = sum(int(episode["workspace_violation_count"]) for episode in selected)
    speed_limit_count = sum(int(episode["speed_limit_count"]) for episode in selected)
    success_rate = len(successes) / float(expected_count)
    return {
        "completed_episodes": len(selected),
        "success_count": len(successes),
        "success_rate": success_rate,
        "collision_count": collision_count,
        "collision_rate": collision_count / float(expected_count),
        "workspace_violation_count": workspace_count,
        "speed_limit_count": speed_limit_count,
        "offset_error": _stats([float(episode["final_offset_error"]) for episode in selected]),
        "relative_speed": _stats([float(episode["final_relative_speed"]) for episode in selected]),
        "success_offset_error": _stats([float(episode["final_offset_error"]) for episode in successes]),
        "success_relative_speed": _stats([float(episode["final_relative_speed"]) for episode in successes]),
        "minimum_center_distance": _stats([float(episode["minimum_center_distance"]) for episode in selected]),
        "convergence_time": _stats(_finite_history_values(selected, "convergence_time")),
        "acceleration_saturation_fraction": _stats(
            [float(episode["acceleration_saturation_fraction"]) for episode in selected]
        ),
        "velocity_command_saturation_fraction": _stats(
            [float(episode["velocity_command_saturation_fraction"]) for episode in selected]
        ),
        "initial_path_min_distance": _stats([float(episode["initial_path_min_distance"]) for episode in selected]),
        "success_hold_completed_count": sum(bool(episode["success_hold_completed"]) for episode in selected),
    }


def _run_scenario(
    env: Any,
    args_cli: argparse.Namespace,
    scenario_name: str,
    mode_name: str,
    fixed_nominal: bool,
    base_motion_cfg: Any,
    base_initial_geometry_cfg: Any,
) -> dict[str, Any]:
    scenario_seed = int(args_cli.seed) + {
        "nominal_fixed_cv": 0,
        "random_constant_velocity": 1000,
        "stress_constant_acceleration": 2000,
        "stress_constant_turn": 3000,
        "stress_piecewise_acceleration": 4000,
    }[scenario_name]
    task = env.unwrapped
    _configure_task_for_scenario(
        task, args_cli, scenario_seed, mode_name, fixed_nominal, base_motion_cfg, base_initial_geometry_cfg
    )
    obs, _ = env.reset(seed=scenario_seed)
    task.get_m4_episode_history(clear=True)
    _assert_all_finite("reset_obs", obs)
    steps = int(task.max_episode_length * args_cli.episodes)
    max_sync = _asset_sync_errors(task)
    max_target_cv_error = 0.0

    for _ in range(steps):
        actions = torch.zeros(env.action_space.shape, device=task.device)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        _assert_all_finite("obs", obs)
        _assert_all_finite("rewards", rewards)
        max_sync = _max_asset_sync_errors(max_sync, _asset_sync_errors(task))
        if mode_name == "ConstantVelocity":
            max_target_cv_error = max(max_target_cv_error, _target_cv_error(task))
        _assert(not bool(terminated.any().item()), f"{scenario_name} terminated early before requested episodes.")
        if terminated.shape != truncated.shape:
            raise RuntimeError("Terminated and truncated tensors have mismatched shapes.")

    history = task.get_m4_episode_history(clear=True)
    summary = _summarize_history(history, args_cli.num_envs * args_cli.episodes)
    diagnostics = task.get_m4_diagnostics()
    finite = bool(diagnostics["finite_check"])
    if mode_name == "ConstantVelocity":
        _assert(max_target_cv_error <= TARGET_CV_TOL, "ConstantVelocity target analytic error too large.")
    _assert(max_sync["target_position"] <= SYNC_POSITION_TOL, "Target asset position sync error too large.")
    _assert(max_sync["target_velocity"] <= SYNC_VELOCITY_TOL, "Target asset velocity sync error too large.")
    _assert(max_sync["ego_position"] <= SYNC_POSITION_TOL, "Ego asset position sync error too large.")
    _assert(max_sync["ego_velocity"] <= SYNC_VELOCITY_TOL, "Ego asset velocity sync error too large.")
    _assert(finite, f"{scenario_name} diagnostics reported non-finite state.")
    _assert(summary["collision_count"] == 0, f"{scenario_name} reported collision risk.")

    return {
        "scenario": scenario_name,
        "mode": mode_name,
        "seed": scenario_seed,
        "num_envs": int(args_cli.num_envs),
        "episodes": int(args_cli.episodes),
        "steps": steps,
        "sim_dt": float(task.physics_dt),
        "decimation": int(task.cfg.decimation),
        "env_step_dt": float(task.step_dt),
        "summary": summary,
        "asset_sync_errors": max_sync,
        "target_analytic_error": max_target_cv_error if mode_name == "ConstantVelocity" else None,
        "finite": finite,
        "diagnostics": diagnostics,
    }


def _assert_acceptance(report: dict[str, Any]) -> dict[str, bool]:
    nominal = report["nominal_fixed_cv"]
    random_cv = report["random_constant_velocity"]
    checks = {
        "nominal_success_rate_100": nominal["summary"]["success_rate"] == 1.0,
        "nominal_collision_count_zero": nominal["summary"]["collision_count"] == 0,
        "nominal_finite": bool(nominal["finite"]),
        "random_success_rate_ge_95": random_cv["summary"]["success_rate"] >= 0.95,
        "random_collision_count_zero": random_cv["summary"]["collision_count"] == 0,
        "random_success_offset_p95_lt_050": random_cv["summary"]["success_offset_error"]["p95"] < 0.50,
        "random_success_relative_speed_p95_lt_030": random_cv["summary"]["success_relative_speed"]["p95"] < 0.30,
        "random_finite": bool(random_cv["finite"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    _assert(not failed, f"M4 acceptance checks failed: {failed}")
    return checks


def main() -> None:
    args_cli = _build_parser().parse_args()
    simulation_app = _launch_app(args_cli)
    env = None
    try:
        env = _make_env(args_cli)
        task = env.unwrapped
        base_motion_cfg = task.cfg.target_motion
        base_initial_geometry_cfg = task.cfg.baseline_initial_geometry
        scenarios = (
            ("nominal_fixed_cv", "ConstantVelocity", True),
            ("random_constant_velocity", "ConstantVelocity", False),
            ("stress_constant_acceleration", "ConstantAcceleration", False),
            ("stress_constant_turn", "ConstantTurn", False),
            ("stress_piecewise_acceleration", "PiecewiseAcceleration", False),
        )
        report = {}
        for scenario_name, mode_name, fixed_nominal in scenarios:
            print(f"[INFO] Running M4 scenario {scenario_name}.", flush=True)
            report[scenario_name] = _run_scenario(
                env,
                args_cli,
                scenario_name,
                mode_name,
                fixed_nominal,
                base_motion_cfg,
                base_initial_geometry_cfg,
            )
        report["acceptance_checks"] = _assert_acceptance(report)
        print(f"[INFO] M4 baseline runtime audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
