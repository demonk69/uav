#!/usr/bin/env python3
"""End-to-end M2 runtime audit under Isaac Lab."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from typing import Any

import torch
from isaaclab.app import AppLauncher


ANALYTIC_POSITION_TOL = 2.0e-3
SYNC_POSITION_TOL = 1.0e-4
SYNC_VELOCITY_TOL = 1.0e-4
EGO_DRIFT_TOL = 1.0e-6
HEIGHT_TOL = 1.0e-6
RESET_STATE_TOL = 1.0e-6

gym = None
parse_env_cfg = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for the UAV rendezvous M2 task.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-Direct-v0", help="Gymnasium task ID.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of environment steps for the main audit.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility checks and main audit.")
    parser.add_argument("--seed_probe", action="store_true", default=False, help=argparse.SUPPRESS)
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


def _make_env(args_cli: argparse.Namespace, seed: int, steps: int | None = None):
    _load_runtime_modules()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = seed
    if hasattr(env_cfg, "target_motion"):
        motion_cfg = env_cfg.target_motion
        cv_probs = (1.0, 0.0, 0.0, 0.0)
        env_cfg.target_motion = replace(
            motion_cfg,
            force_mode_cycle_on_reset=False,
            train=replace(motion_cfg.train, mode_probabilities=cv_probs),
            validation=replace(motion_cfg.validation, mode_probabilities=cv_probs),
            test=replace(motion_cfg.test, mode_probabilities=cv_probs),
        )
        env_cfg.target_motion_split = "train"
    if steps is not None:
        audit_duration_s = (steps + 10) * env_cfg.sim.dt * env_cfg.decimation
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), audit_duration_s)
    return gym.make(args_cli.task, cfg=env_cfg)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_all_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _assert(bool(torch.isfinite(value).all().item()), f"Non-finite tensor detected in {name}.")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_all_finite(f"{name}.{key}", item)


def _clone_task_state(task: Any) -> dict[str, torch.Tensor]:
    return {
        "p_ego_w": task.p_ego_w.detach().clone(),
        "v_ego_w": task.v_ego_w.detach().clone(),
        "p_target_initial_w": task.p_target_initial_w.detach().clone(),
        "p_target_w": task.p_target_w.detach().clone(),
        "v_target_w": task.v_target_w.detach().clone(),
        "p_rel_w": task.p_rel_w.detach().clone(),
        "v_rel_w": task.v_rel_w.detach().clone(),
        "e_offset_w": task.e_offset_w.detach().clone(),
        "target_elapsed_time": task.target_elapsed_time.detach().clone(),
    }


def _target_asset_local_position(task: Any) -> torch.Tensor:
    return task.target.data.root_pos_w - task.scene.env_origins


def _ego_asset_local_position(task: Any) -> torch.Tensor:
    return task.ego.data.root_pos_w - task.scene.env_origins


def _asset_sync_errors(task: Any) -> dict[str, float]:
    target_position_error = torch.max(torch.abs(_target_asset_local_position(task) - task.p_target_w))
    target_velocity_error = torch.max(torch.abs(task.target.data.root_lin_vel_w - task.v_target_w))
    ego_position_error = torch.max(torch.abs(_ego_asset_local_position(task) - task.p_ego_w))
    ego_velocity_error = torch.max(torch.abs(task.ego.data.root_lin_vel_w - task.v_ego_w))
    return {
        "target_position": float(target_position_error.item()),
        "target_velocity": float(target_velocity_error.item()),
        "ego_position": float(ego_position_error.item()),
        "ego_velocity": float(ego_velocity_error.item()),
    }


def _state_to_lists(state: dict[str, torch.Tensor]) -> dict[str, list]:
    return {key: value.detach().cpu().tolist() for key, value in state.items()}


def _seed_probe_report(args_cli: argparse.Namespace, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    state_json = json.dumps(_state_to_lists(state), sort_keys=True)
    target_initial = state["p_target_initial_w"]
    target_velocity = state["v_target_w"]
    p_rel_w = state["p_rel_w"]
    return {
        "seed": int(args_cli.seed),
        "num_envs": int(args_cli.num_envs),
        "state_sha256": hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
        "target_positions_independent": bool(torch.unique(target_initial, dim=0).shape[0] > 1),
        "target_velocities_independent": bool(torch.unique(target_velocity, dim=0).shape[0] > 1),
        "min_reset_relative_distance": float(torch.linalg.norm(p_rel_w, dim=1).min().item()),
    }


def _collect_reset_state(env: Any, seed: int) -> dict[str, torch.Tensor]:
    obs, _ = env.reset(seed=seed)
    _assert_all_finite("seed_audit_obs", obs)
    return _clone_task_state(env.unwrapped)


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(left[key], right[key]) for key in left)


def _state_any_different(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(left[key], right[key]) for key in left)


def _run_seed_audit(args_cli: argparse.Namespace, env: Any) -> dict[str, Any]:
    same_seed_a = _collect_reset_state(env, args_cli.seed)
    same_seed_b = _collect_reset_state(env, args_cli.seed)
    different_seed = _collect_reset_state(env, args_cli.seed + 1)

    target_initial = same_seed_a["p_target_initial_w"]
    target_velocity = same_seed_a["v_target_w"]
    target_positions_independent = torch.unique(target_initial, dim=0).shape[0] > 1
    target_velocities_independent = torch.unique(target_velocity, dim=0).shape[0] > 1

    same_seed_reproduced = _state_equal(same_seed_a, same_seed_b)
    different_seed_changed = _state_any_different(same_seed_a, different_seed)
    independent_env_randomization = bool(target_positions_independent and target_velocities_independent)

    _assert(same_seed_reproduced, "Same-seed independent env instances did not reproduce initial state.")
    _assert(different_seed_changed, "Different seed did not change any sampled initial state.")
    _assert(independent_env_randomization, "All envs received identical target initial position or velocity.")

    return {
        "method": "single_env_reseed_runtime_check",
        "same_seed_reproduced": same_seed_reproduced,
        "different_seed_changed": different_seed_changed,
        "independent_env_randomization": independent_env_randomization,
    }


def _check_partial_reset(task: Any) -> dict[str, Any]:
    _assert(task.num_envs > 7, "Partial reset audit requires at least 8 environments.")
    env_ids = torch.tensor([1, 3, 7], dtype=torch.long, device=task.device)
    untouched_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    untouched_mask[env_ids] = False
    before = _clone_task_state(task)

    task._reset_idx(env_ids)
    after = _clone_task_state(task)

    max_untouched_delta = 0.0
    for key, before_value in before.items():
        delta = torch.max(torch.abs(after[key][untouched_mask] - before_value[untouched_mask]))
        max_untouched_delta = max(max_untouched_delta, float(delta.item()))

    selected_elapsed_max = float(torch.max(torch.abs(task.target_elapsed_time[env_ids])).item())
    sync_errors = _asset_sync_errors(task)
    selected_target_sync = torch.max(torch.abs(_target_asset_local_position(task)[env_ids] - task.p_target_w[env_ids]))
    selected_ego_sync = torch.max(torch.abs(_ego_asset_local_position(task)[env_ids] - task.p_ego_w[env_ids]))
    selected_position_sync_error = max(float(selected_target_sync.item()), float(selected_ego_sync.item()))

    _assert(max_untouched_delta <= RESET_STATE_TOL, "Partial reset changed an unselected env state.")
    _assert(selected_elapsed_max <= RESET_STATE_TOL, "Partial reset did not zero selected target elapsed time.")
    _assert(selected_position_sync_error <= SYNC_POSITION_TOL, "Partial reset did not sync selected entity poses.")

    return {
        "passed": True,
        "reset_env_ids": env_ids.detach().cpu().tolist(),
        "max_untouched_state_delta": max_untouched_delta,
        "selected_elapsed_time_max": selected_elapsed_max,
        "selected_position_sync_error": selected_position_sync_error,
        "post_reset_sync_errors": sync_errors,
    }


def _run_main_audit(args_cli: argparse.Namespace, env: Any, seed_audit: dict[str, Any]) -> dict[str, Any]:
    obs, _ = env.reset(seed=args_cli.seed)
    task = env.unwrapped
    _assert_all_finite("reset_obs", obs)

    ego_collision_enabled = task.cfg.ego_cfg.spawn.collision_props.collision_enabled
    target_collision_enabled = task.cfg.target_cfg.spawn.collision_props.collision_enabled
    initial = _clone_task_state(task)
    initial_relative_distance = torch.linalg.norm(task.p_rel_w, dim=1)
    min_relative_distance = float(initial_relative_distance.min().item())
    _assert(min_relative_distance > task.cfg.d_safe, "Reset produced a center distance <= d_safe.")

    for _ in range(args_cli.steps):
        actions = torch.zeros(env.action_space.shape, device=task.device)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        _assert_all_finite("obs", obs)
        _assert_all_finite("rewards", rewards)
        _assert(not bool(terminated.any().item()), "Main audit terminated before completing requested steps.")
        _assert(not bool(truncated.any().item()), "Main audit timed out before completing requested steps.")
        min_relative_distance = min(min_relative_distance, float(torch.linalg.norm(task.p_rel_w, dim=1).min().item()))

    elapsed = args_cli.steps * task.step_dt
    expected_target_position = initial["p_target_initial_w"] + initial["v_target_w"] * elapsed
    expected_target_position[:, 2] = initial["p_target_initial_w"][:, 2]
    analytic_position_error = torch.max(torch.abs(task.p_target_w - expected_target_position))

    asset_sync = _asset_sync_errors(task)
    ego_position_drift = torch.max(torch.abs(task.p_ego_w - initial["p_ego_w"]))
    ego_velocity_drift = torch.max(torch.abs(task.v_ego_w - initial["v_ego_w"]))
    ego_max_drift = max(float(ego_position_drift.item()), float(ego_velocity_drift.item()))
    fixed_height_error = torch.max(torch.abs(task.p_target_w[:, 2] - task.p_target_initial_w[:, 2]))

    actual_average_velocity = (task.p_target_w - initial["p_target_initial_w"]) / elapsed
    target_actual_average_speed = float(torch.linalg.norm(actual_average_velocity, dim=1).mean().item())
    target_truth_average_speed = float(torch.linalg.norm(task.v_target_w, dim=1).mean().item())

    _assert(float(analytic_position_error.item()) <= ANALYTIC_POSITION_TOL, "Target analytic position error too large.")
    _assert(asset_sync["target_position"] <= SYNC_POSITION_TOL, "Target asset position is not synchronized.")
    _assert(asset_sync["target_velocity"] <= SYNC_VELOCITY_TOL, "Target asset velocity is not synchronized.")
    _assert(asset_sync["ego_position"] <= SYNC_POSITION_TOL, "Ego asset position is not synchronized.")
    _assert(asset_sync["ego_velocity"] <= SYNC_VELOCITY_TOL, "Ego asset velocity is not synchronized.")
    _assert(ego_max_drift <= EGO_DRIFT_TOL, "Ego drifted during main audit.")
    _assert(float(fixed_height_error.item()) <= HEIGHT_TOL, "Target height was not fixed.")
    _assert(min_relative_distance > task.cfg.d_safe, "Runtime center distance became <= d_safe.")
    _assert_all_finite("task_state", _clone_task_state(task))
    _assert_all_finite("target_asset_pos", task.target.data.root_pos_w)
    _assert_all_finite("target_asset_vel", task.target.data.root_lin_vel_w)
    _assert_all_finite("ego_asset_pos", task.ego.data.root_pos_w)
    _assert_all_finite("ego_asset_vel", task.ego.data.root_lin_vel_w)

    partial_reset = _check_partial_reset(task)

    return {
        "passed": True,
        "truth_authority": "task_tensors",
        "rigid_object_role": "synchronized_state_and_visualization_carrier",
        "num_envs": int(task.num_envs),
        "steps": int(args_cli.steps),
        "seed": int(args_cli.seed),
        "sim_dt": float(task.physics_dt),
        "decimation": int(task.cfg.decimation),
        "env_step_dt": float(task.step_dt),
        "target_actual_average_speed": target_actual_average_speed,
        "target_truth_average_speed": target_truth_average_speed,
        "max_analytic_position_error": float(analytic_position_error.item()),
        "asset_sync_errors": asset_sync,
        "ego_max_drift": ego_max_drift,
        "fixed_height_max_error": float(fixed_height_error.item()),
        "min_relative_distance": min_relative_distance,
        "d_safe": float(task.cfg.d_safe),
        "collision_enabled": {"ego": ego_collision_enabled, "target": target_collision_enabled},
        "no_contact_verified_by_center_distance": True,
        "no_physics_disturbance_verified_by_sync_and_ego_drift": True,
        "partial_reset": partial_reset,
        "seed_audit": seed_audit,
        "finite": True,
    }


def main() -> None:
    args_cli = _build_parser().parse_args()
    simulation_app = _launch_app(args_cli)
    env = None

    try:
        env = _make_env(args_cli, args_cli.seed, steps=args_cli.steps)
        if args_cli.seed_probe:
            report = _seed_probe_report(args_cli, _collect_reset_state(env, args_cli.seed))
            print(f"[INFO] M2 seed probe: {json.dumps(report, sort_keys=True)}", flush=True)
            return

        print("[INFO] Running M2 seed audit by reseeding one runtime env instance.", flush=True)
        seed_audit = _run_seed_audit(args_cli, env)
        print("[INFO] M2 seed audit passed. Running main runtime audit.", flush=True)
        report = _run_main_audit(args_cli, env, seed_audit)
        print(f"[INFO] M2 runtime audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
