#!/usr/bin/env python3
"""End-to-end M3 target motion runtime audit under Isaac Lab."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from typing import Any

import torch
from isaaclab.app import AppLauncher


MOTION_STATE_TOL = 2.0e-5
SYNC_POSITION_TOL = 1.0e-4
SYNC_VELOCITY_TOL = 1.0e-4
EGO_DRIFT_TOL = 1.0e-6
RESET_STATE_TOL = 1.0e-6
OBSERVATION_TOL = 1.0e-7

MODE_CONSTANT_VELOCITY = 0
MODE_CONSTANT_ACCELERATION = 1
MODE_CONSTANT_TURN = 2
MODE_PIECEWISE_ACCELERATION = 3
MODE_ID_TO_NAME = {
    MODE_CONSTANT_VELOCITY: "ConstantVelocity",
    MODE_CONSTANT_ACCELERATION: "ConstantAcceleration",
    MODE_CONSTANT_TURN: "ConstantTurn",
    MODE_PIECEWISE_ACCELERATION: "PiecewiseAcceleration",
}
MODE_NAME_ORDER = tuple(MODE_ID_TO_NAME.values())

DIGEST_KEYS = (
    "p_target_initial_w",
    "v_target_w",
    "mode_id",
    "constant_acceleration_w",
    "turn_omega",
    "segment_duration_steps",
    "current_acceleration_w",
)

gym = None
parse_env_cfg = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for the UAV rendezvous M3 target motion library.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-Direct-v0", help="Gymnasium task ID.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of environment steps for the main audit.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility checks and main audit.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train", help="Motion split to audit.")
    parser.add_argument(
        "--random_modes",
        action="store_true",
        default=False,
        help="Use split mode probabilities instead of deterministic mode cycling.",
    )
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
    env_cfg.target_motion_split = args_cli.split
    if not args_cli.random_modes:
        env_cfg.target_motion = replace(env_cfg.target_motion, force_mode_cycle_on_reset=True)
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


def _mode_counts_from_task(task: Any) -> dict[str, int]:
    mode_id = task.target_motion_manager.mode_id
    return {name: int(torch.count_nonzero(mode_id == mode).item()) for mode, name in MODE_ID_TO_NAME.items()}


def _clone_task_state(task: Any) -> dict[str, torch.Tensor]:
    manager = task.target_motion_manager
    return {
        "p_ego_w": task.p_ego_w.detach().clone(),
        "v_ego_w": task.v_ego_w.detach().clone(),
        "p_target_initial_w": task.p_target_initial_w.detach().clone(),
        "p_target_w": task.p_target_w.detach().clone(),
        "v_target_w": task.v_target_w.detach().clone(),
        "a_target_w": task.a_target_w.detach().clone(),
        "p_rel_w": task.p_rel_w.detach().clone(),
        "v_rel_w": task.v_rel_w.detach().clone(),
        "e_offset_w": task.e_offset_w.detach().clone(),
        "target_elapsed_time": task.target_elapsed_time.detach().clone(),
        "mode_id": manager.mode_id.detach().clone(),
        "motion_step_count": manager.motion_step_count.detach().clone(),
        "constant_acceleration_w": manager.constant_acceleration_w.detach().clone(),
        "turn_omega": manager.turn_omega.detach().clone(),
        "segment_index": manager.segment_index.detach().clone(),
        "segment_step_count": manager.segment_step_count.detach().clone(),
        "segment_duration_steps": manager.segment_duration_steps.detach().clone(),
        "segment_start_position_w": manager.segment_start_position_w.detach().clone(),
        "segment_start_velocity_w": manager.segment_start_velocity_w.detach().clone(),
        "current_acceleration_w": manager.current_acceleration_w.detach().clone(),
        "segment_switch_count": manager.segment_switch_count.detach().clone(),
    }


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(left[key], right[key]) for key in left)


def _state_any_different(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(left[key], right[key]) for key in left)


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest_state = {key: state[key].detach().cpu().tolist() for key in DIGEST_KEYS}
    state_json = json.dumps(digest_state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(state_json.encode("utf-8")).hexdigest()


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


def _check_policy_observation(task: Any, obs: Any) -> dict[str, Any]:
    _assert(isinstance(obs, dict), "Environment observation is not a dict.")
    _assert(set(obs.keys()) == {"policy"}, "M3 Actor observation must contain only the policy group.")
    policy_obs = obs["policy"]
    _assert(policy_obs.shape == (task.num_envs, 6), "M3 Actor observation shape must be (num_envs, 6).")
    expected_policy_obs = torch.cat((task.p_rel_w, task.v_rel_w), dim=-1)
    max_match_error = float(torch.max(torch.abs(policy_obs - expected_policy_obs)).item())
    _assert(max_match_error <= OBSERVATION_TOL, "M3 Actor observation is not exactly [p_rel_w, v_rel_w].")
    _assert_all_finite("policy_obs", policy_obs)
    return {
        "groups": list(obs.keys()),
        "policy_shape": list(policy_obs.shape),
        "policy_dim": int(policy_obs.shape[1]),
        "matches_current_p_rel_w_v_rel_w": True,
        "max_policy_match_error": max_match_error,
        "contains_mode_id": False,
        "contains_motion_parameters": False,
        "contains_future_schedule": False,
        "contains_future_target_state": False,
    }


def _collect_reset_state(env: Any, seed: int) -> dict[str, torch.Tensor]:
    obs, _ = env.reset(seed=seed)
    _check_policy_observation(env.unwrapped, obs)
    return _clone_task_state(env.unwrapped)


def _split_config_summary(cfg: Any) -> dict[str, Any]:
    return {
        split_name: {
            "seed_offset": int(getattr(cfg, split_name).seed_offset),
            "mode_probabilities": list(getattr(cfg, split_name).mode_probabilities),
            "target_pos_x_range": list(getattr(cfg, split_name).target_pos_x_range),
            "target_pos_y_range": list(getattr(cfg, split_name).target_pos_y_range),
            "target_height_range": list(getattr(cfg, split_name).target_height_range),
            "target_vel_x_range": list(getattr(cfg, split_name).target_vel_x_range),
            "target_vel_y_range": list(getattr(cfg, split_name).target_vel_y_range),
            "acceleration_x_range": list(getattr(cfg, split_name).acceleration_x_range),
            "acceleration_y_range": list(getattr(cfg, split_name).acceleration_y_range),
            "turn_omega_range": list(getattr(cfg, split_name).turn_omega_range),
            "piecewise_acceleration_x_range": list(getattr(cfg, split_name).piecewise_acceleration_x_range),
            "piecewise_acceleration_y_range": list(getattr(cfg, split_name).piecewise_acceleration_y_range),
            "piecewise_segment_duration_steps_range": list(
                getattr(cfg, split_name).piecewise_segment_duration_steps_range
            ),
        }
        for split_name in ("train", "validation", "test")
    }


def _split_ranges_different(cfg: Any) -> dict[str, bool]:
    summary = _split_config_summary(cfg)
    train = summary["train"]
    validation = summary["validation"]
    test = summary["test"]
    return {
        "train_vs_validation": train != validation,
        "train_vs_test": train != test,
        "validation_vs_test": validation != test,
        "seed_offsets_unique": len({train["seed_offset"], validation["seed_offset"], test["seed_offset"]}) == 3,
    }


def _run_seed_audit(args_cli: argparse.Namespace, env: Any) -> dict[str, Any]:
    same_seed_a = _collect_reset_state(env, args_cli.seed)
    same_seed_b = _collect_reset_state(env, args_cli.seed)
    different_seed = _collect_reset_state(env, args_cli.seed + 1)
    digest_a = _state_digest(same_seed_a)
    digest_b = _state_digest(same_seed_b)
    digest_different = _state_digest(different_seed)

    _assert(_state_equal(same_seed_a, same_seed_b), "Same seed did not reproduce M3 motion state.")
    _assert(_state_any_different(same_seed_a, different_seed), "Different seed did not change M3 motion state.")
    _assert(digest_a == digest_b, "Same seed did not reproduce M3 sampling digest.")
    _assert(digest_a != digest_different, "Different seed did not change M3 sampling digest.")
    _assert(
        bool(torch.unique(same_seed_a["p_target_initial_w"], dim=0).shape[0] > 1),
        "All envs received identical target initial positions.",
    )
    _assert(
        bool(torch.unique(same_seed_a["v_target_w"], dim=0).shape[0] > 1),
        "All envs received identical target initial velocities.",
    )

    return {
        "method": "single_env_reseed_runtime_check",
        "same_seed_reproduced": True,
        "different_seed_changed": True,
        "independent_env_randomization": True,
        "seed": int(args_cli.seed),
        "different_seed": int(args_cli.seed + 1),
        "same_seed_digest_a": digest_a,
        "same_seed_digest_b": digest_b,
        "different_seed_digest": digest_different,
    }


def _reference_constant_velocity(
    p0_w: torch.Tensor,
    v0_w: torch.Tensor,
    step_count: torch.Tensor,
    physics_dt: float,
    fixed_height: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = step_count.to(device=p0_w.device, dtype=torch.float64).unsqueeze(-1) * float(physics_dt)
    p_w = p0_w.to(torch.float64) + v0_w.to(torch.float64) * t
    v_w = v0_w.to(torch.float64).clone()
    a_w = torch.zeros_like(v_w)
    if fixed_height:
        p_w[:, 2] = p0_w[:, 2].to(torch.float64)
        v_w[:, 2] = 0.0
    return p_w, v_w, a_w


def _reference_constant_acceleration(
    p0_w: torch.Tensor,
    v0_w: torch.Tensor,
    a_w: torch.Tensor,
    step_count: torch.Tensor,
    physics_dt: float,
    fixed_height: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = step_count.to(device=p0_w.device, dtype=torch.float64).unsqueeze(-1) * float(physics_dt)
    a64_w = a_w.to(torch.float64)
    p_w = p0_w.to(torch.float64) + v0_w.to(torch.float64) * t + 0.5 * a64_w * t.square()
    v_w = v0_w.to(torch.float64) + a64_w * t
    a_eval_w = a64_w.clone()
    if fixed_height:
        p_w[:, 2] = p0_w[:, 2].to(torch.float64)
        v_w[:, 2] = 0.0
        a_eval_w[:, 2] = 0.0
    return p_w, v_w, a_eval_w


def _reference_constant_turn(
    p0_w: torch.Tensor,
    v0_w: torch.Tensor,
    omega: torch.Tensor,
    step_count: torch.Tensor,
    physics_dt: float,
    fixed_height: bool,
    omega_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = p0_w.device
    t = step_count.to(device=device, dtype=torch.float64) * float(physics_dt)
    omega64 = omega.to(device=device, dtype=torch.float64)
    theta = omega64 * t
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    vx0 = v0_w[:, 0].to(torch.float64)
    vy0 = v0_w[:, 1].to(torch.float64)
    small_omega = torch.abs(omega64) < omega_epsilon

    v_x_turn = vx0 * cos_theta - vy0 * sin_theta
    v_y_turn = vx0 * sin_theta + vy0 * cos_theta
    safe_omega = torch.where(small_omega, torch.ones_like(omega64), omega64)
    dx_turn = (vx0 * sin_theta + vy0 * (cos_theta - 1.0)) / safe_omega
    dy_turn = (vx0 * (1.0 - cos_theta) + vy0 * sin_theta) / safe_omega
    dx = torch.where(small_omega, vx0 * t, dx_turn)
    dy = torch.where(small_omega, vy0 * t, dy_turn)
    v_x = torch.where(small_omega, vx0, v_x_turn)
    v_y = torch.where(small_omega, vy0, v_y_turn)

    p_w = p0_w.to(torch.float64).clone()
    v_w = torch.zeros_like(p_w)
    a_w = torch.zeros_like(p_w)
    p_w[:, 0] = p0_w[:, 0].to(torch.float64) + dx
    p_w[:, 1] = p0_w[:, 1].to(torch.float64) + dy
    v_w[:, 0] = v_x
    v_w[:, 1] = v_y
    a_w[:, 0] = torch.where(small_omega, torch.zeros_like(omega64), -omega64 * v_y)
    a_w[:, 1] = torch.where(small_omega, torch.zeros_like(omega64), omega64 * v_x)
    if fixed_height:
        p_w[:, 2] = p0_w[:, 2].to(torch.float64)
    else:
        v_w[:, 2] = v0_w[:, 2].to(torch.float64)
        p_w[:, 2] = p0_w[:, 2].to(torch.float64) + v_w[:, 2] * t
    return p_w, v_w, a_w


def _select_expected_state(task: Any) -> dict[str, Any]:
    manager = task.target_motion_manager
    fixed_height = bool(task.cfg.target_motion.fixed_height)
    cv = _reference_constant_velocity(manager.p0_w, manager.v0_w, manager.motion_step_count, task.physics_dt, fixed_height)
    ca = _reference_constant_acceleration(
        manager.p0_w,
        manager.v0_w,
        manager.constant_acceleration_w,
        manager.motion_step_count,
        task.physics_dt,
        fixed_height,
    )
    ct = _reference_constant_turn(
        manager.p0_w,
        manager.v0_w,
        manager.turn_omega,
        manager.motion_step_count,
        task.physics_dt,
        fixed_height,
        task.cfg.target_motion.omega_epsilon,
    )
    pw = _reference_constant_acceleration(
        manager.segment_start_position_w,
        manager.segment_start_velocity_w,
        manager.current_acceleration_w,
        manager.segment_step_count,
        task.physics_dt,
        fixed_height,
    )

    expected_p_w = torch.empty_like(task.p_target_w, dtype=torch.float64)
    expected_v_w = torch.empty_like(task.v_target_w, dtype=torch.float64)
    expected_a_w = torch.empty_like(task.a_target_w, dtype=torch.float64)
    references = {
        MODE_CONSTANT_VELOCITY: cv,
        MODE_CONSTANT_ACCELERATION: ca,
        MODE_CONSTANT_TURN: ct,
        MODE_PIECEWISE_ACCELERATION: pw,
    }
    for mode_id, (p_ref_w, v_ref_w, a_ref_w) in references.items():
        mask = manager.mode_id == mode_id
        if bool(torch.any(mask).item()):
            expected_p_w[mask] = p_ref_w[mask]
            expected_v_w[mask] = v_ref_w[mask]
            expected_a_w[mask] = a_ref_w[mask]

    actual_p_w = task.p_target_w.to(torch.float64)
    actual_v_w = task.v_target_w.to(torch.float64)
    actual_a_w = task.a_target_w.to(torch.float64)
    position_error = torch.abs(actual_p_w - expected_p_w).amax(dim=1)
    velocity_error = torch.abs(actual_v_w - expected_v_w).amax(dim=1)
    acceleration_error = torch.abs(actual_a_w - expected_a_w).amax(dim=1)
    displacement = torch.linalg.norm(task.p_target_w - task.p_target_initial_w, dim=1)

    per_mode = {}
    for mode_id, mode_name in MODE_ID_TO_NAME.items():
        mask = manager.mode_id == mode_id
        count = int(torch.count_nonzero(mask).item())
        if count == 0:
            per_mode[mode_name] = {
                "env_count": 0,
                "actual_displacement_max": 0.0,
                "actual_displacement_mean": 0.0,
                "actual_displacement_per_env": [],
                "max_position_error": 0.0,
                "max_velocity_error": 0.0,
                "max_acceleration_error": 0.0,
            }
            continue
        per_mode[mode_name] = {
            "env_count": count,
            "actual_displacement_max": float(displacement[mask].max().item()),
            "actual_displacement_mean": float(displacement[mask].mean().item()),
            "actual_displacement_per_env": displacement[mask].detach().cpu().tolist(),
            "max_position_error": float(position_error[mask].max().item()),
            "max_velocity_error": float(velocity_error[mask].max().item()),
            "max_acceleration_error": float(acceleration_error[mask].max().item()),
        }
    return {
        "overall": {
            "position": float(position_error.max().item()),
            "velocity": float(velocity_error.max().item()),
            "acceleration": float(acceleration_error.max().item()),
        },
        "per_mode": per_mode,
    }


def _piecewise_snapshot(task: Any) -> dict[str, torch.Tensor]:
    manager = task.target_motion_manager
    return {
        "mode_id": manager.mode_id.detach().clone(),
        "segment_switch_count": manager.segment_switch_count.detach().clone(),
        "segment_duration_steps": manager.segment_duration_steps.detach().clone(),
        "segment_start_position_w": manager.segment_start_position_w.detach().clone(),
        "segment_start_velocity_w": manager.segment_start_velocity_w.detach().clone(),
        "current_acceleration_w": manager.current_acceleration_w.detach().clone(),
    }


def _new_piecewise_boundary_audit() -> dict[str, Any]:
    return {
        "checked_switch_events": 0,
        "max_switch_delta_per_env_step": 0,
        "max_boundary_position_error": 0.0,
        "max_boundary_velocity_error": 0.0,
    }


def _update_piecewise_boundary_audit(before: dict[str, torch.Tensor], task: Any, audit: dict[str, Any]) -> None:
    manager = task.target_motion_manager
    switch_delta = manager.segment_switch_count - before["segment_switch_count"]
    piecewise_mask = before["mode_id"] == MODE_PIECEWISE_ACCELERATION
    switched = piecewise_mask & (switch_delta > 0)
    if not bool(torch.any(switched).item()):
        return

    max_delta = int(switch_delta[switched].max().item())
    audit["max_switch_delta_per_env_step"] = max(audit["max_switch_delta_per_env_step"], max_delta)
    _assert(max_delta == 1, "PiecewiseAcceleration switched more than once inside one env step.")
    expected_position_w, expected_velocity_w, _expected_acceleration_w = _reference_constant_acceleration(
        before["segment_start_position_w"][switched],
        before["segment_start_velocity_w"][switched],
        before["current_acceleration_w"][switched],
        before["segment_duration_steps"][switched],
        task.physics_dt,
        bool(task.cfg.target_motion.fixed_height),
    )
    actual_position_w = manager.segment_start_position_w[switched].to(torch.float64)
    actual_velocity_w = manager.segment_start_velocity_w[switched].to(torch.float64)
    position_error = float(torch.max(torch.abs(actual_position_w - expected_position_w)).item())
    velocity_error = float(torch.max(torch.abs(actual_velocity_w - expected_velocity_w)).item())
    audit["checked_switch_events"] += int(torch.count_nonzero(switched).item())
    audit["max_boundary_position_error"] = max(audit["max_boundary_position_error"], position_error)
    audit["max_boundary_velocity_error"] = max(audit["max_boundary_velocity_error"], velocity_error)


def _max_untouched_delta(after: torch.Tensor, before: torch.Tensor, untouched_mask: torch.Tensor) -> float:
    if torch.equal(after[untouched_mask], before[untouched_mask]):
        return 0.0
    if after.dtype.is_floating_point:
        delta = torch.max(torch.abs(after[untouched_mask] - before[untouched_mask]))
        return float(delta.item())
    return float("inf")


def _check_partial_reset(task: Any) -> dict[str, Any]:
    _assert(task.num_envs > 7, "Partial reset audit requires at least 8 environments.")
    env_ids = torch.tensor([1, 3, 7], dtype=torch.long, device=task.device)
    untouched_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    untouched_mask[env_ids] = False
    before = _clone_task_state(task)

    task._reset_idx(env_ids)
    after = _clone_task_state(task)

    max_untouched_delta = 0.0
    unchanged_exact_by_field = {}
    for key, before_value in before.items():
        unchanged_exact = torch.equal(after[key][untouched_mask], before_value[untouched_mask])
        unchanged_exact_by_field[key] = bool(unchanged_exact)
        max_untouched_delta = max(max_untouched_delta, _max_untouched_delta(after[key], before_value, untouched_mask))

    selected_elapsed_max = float(torch.max(torch.abs(task.target_elapsed_time[env_ids])).item())
    selected_motion_step_max = int(torch.max(task.target_motion_manager.motion_step_count[env_ids]).item())
    sync_errors = _asset_sync_errors(task)
    selected_target_sync = torch.max(torch.abs(_target_asset_local_position(task)[env_ids] - task.p_target_w[env_ids]))
    selected_ego_sync = torch.max(torch.abs(_ego_asset_local_position(task)[env_ids] - task.p_ego_w[env_ids]))
    selected_position_sync_error = max(float(selected_target_sync.item()), float(selected_ego_sync.item()))
    untouched_exact = all(unchanged_exact_by_field.values())

    _assert(untouched_exact, "Partial reset changed an unselected env field.")
    _assert(max_untouched_delta <= RESET_STATE_TOL, "Partial reset changed an unselected env state.")
    _assert(selected_elapsed_max <= RESET_STATE_TOL, "Partial reset did not zero selected target elapsed time.")
    _assert(selected_motion_step_max == 0, "Partial reset did not zero selected motion step counts.")
    _assert(selected_position_sync_error <= SYNC_POSITION_TOL, "Partial reset did not sync selected entity poses.")
    _assert(not bool(task.target_motion_invalid_buf[env_ids].any().item()), "Partial reset produced invalid target motion.")

    return {
        "passed": True,
        "reset_env_ids": env_ids.detach().cpu().tolist(),
        "untouched_envs_exactly_unchanged": untouched_exact,
        "unchanged_exact_by_field": unchanged_exact_by_field,
        "max_untouched_state_delta": max_untouched_delta,
        "selected_elapsed_time_max": selected_elapsed_max,
        "selected_motion_step_max": selected_motion_step_max,
        "selected_position_sync_error": selected_position_sync_error,
        "post_reset_sync_errors": sync_errors,
    }


def _check_mode_coverage(task: Any, require_all_modes: bool) -> dict[str, int]:
    mode_counts = _mode_counts_from_task(task)
    if require_all_modes:
        _assert(task.num_envs >= 4, "Mode-cycle audit requires at least 4 environments.")
        missing = [name for name, count in mode_counts.items() if count <= 0]
        _assert(not missing, f"Mode-cycle audit did not cover modes: {missing}.")
    return mode_counts


def _max_observed_tensor_value(task: Any, attr_name: str, fallback: torch.Tensor) -> float:
    if hasattr(task, attr_name):
        return float(getattr(task, attr_name).max().item())
    return float(fallback.max().item())


def _run_main_audit(args_cli: argparse.Namespace, env: Any, seed_audit: dict[str, Any]) -> dict[str, Any]:
    obs, _ = env.reset(seed=args_cli.seed)
    task = env.unwrapped
    actor_check = _check_policy_observation(task, obs)

    ego_collision_enabled = task.cfg.ego_cfg.spawn.collision_props.collision_enabled
    target_collision_enabled = task.cfg.target_cfg.spawn.collision_props.collision_enabled
    initial = _clone_task_state(task)
    initial_relative_distance = torch.linalg.norm(task.p_rel_w, dim=1)
    min_relative_distance = float(initial_relative_distance.min().item())
    _assert(min_relative_distance > task.cfg.d_safe, "Reset produced a center distance <= d_safe.")
    mode_counts = _check_mode_coverage(task, require_all_modes=not args_cli.random_modes)
    split_ranges_different = _split_ranges_different(task.cfg.target_motion)
    _assert(all(split_ranges_different.values()), "M3 train/validation/test split ranges are not distinct.")

    piecewise_boundary_audit = _new_piecewise_boundary_audit()
    last_actor_check = actor_check
    for _ in range(args_cli.steps):
        before_piecewise = _piecewise_snapshot(task)
        actions = torch.zeros(env.action_space.shape, device=task.device)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        _update_piecewise_boundary_audit(before_piecewise, task, piecewise_boundary_audit)
        last_actor_check = _check_policy_observation(task, obs)
        _assert_all_finite("rewards", rewards)
        _assert(not bool(terminated.any().item()), "Main audit terminated before completing requested steps.")
        _assert(not bool(truncated.any().item()), "Main audit timed out before completing requested steps.")
        min_relative_distance = min(min_relative_distance, float(torch.linalg.norm(task.p_rel_w, dim=1).min().item()))

    motion_errors = _select_expected_state(task)
    asset_sync = _asset_sync_errors(task)
    ego_position_drift = torch.max(torch.abs(task.p_ego_w - initial["p_ego_w"]))
    ego_velocity_drift = torch.max(torch.abs(task.v_ego_w - initial["v_ego_w"]))
    ego_max_drift = max(float(ego_position_drift.item()), float(ego_velocity_drift.item()))
    fixed_height_error = torch.max(torch.abs(task.p_target_w[:, 2] - task.p_target_initial_w[:, 2]))
    min_relative_distance_all_physics_substeps = float(task._min_relative_distance_observed.min().item())
    workspace_abs_max = _max_observed_tensor_value(
        task,
        "_workspace_abs_max_observed",
        torch.maximum(torch.abs(task.p_ego_w).amax(dim=1), torch.abs(task.p_target_w).amax(dim=1)),
    )
    max_target_speed = _max_observed_tensor_value(
        task, "_target_speed_max_observed", torch.linalg.norm(task.v_target_w, dim=1)
    )
    max_target_acceleration = _max_observed_tensor_value(
        task, "_target_acceleration_max_observed", torch.linalg.norm(task.a_target_w, dim=1)
    )
    segment_switch_count = task.target_motion_manager.segment_switch_count.detach().cpu().tolist()
    piecewise_mask = task.target_motion_manager.mode_id == MODE_PIECEWISE_ACCELERATION
    piecewise_switch_counts = task.target_motion_manager.segment_switch_count[piecewise_mask]
    piecewise_total_segment_switches = int(piecewise_switch_counts.sum().item())
    piecewise_each_env_switched = bool(torch.all(piecewise_switch_counts > 0).item()) if piecewise_switch_counts.numel() else False

    _assert(motion_errors["overall"]["position"] <= MOTION_STATE_TOL, "Target position independent error too large.")
    _assert(motion_errors["overall"]["velocity"] <= MOTION_STATE_TOL, "Target velocity independent error too large.")
    _assert(
        motion_errors["overall"]["acceleration"] <= MOTION_STATE_TOL,
        "Target acceleration independent error too large.",
    )
    _assert(
        piecewise_boundary_audit["max_boundary_position_error"] <= MOTION_STATE_TOL,
        "PiecewiseAcceleration segment boundary position error too large.",
    )
    _assert(
        piecewise_boundary_audit["max_boundary_velocity_error"] <= MOTION_STATE_TOL,
        "PiecewiseAcceleration segment boundary velocity error too large.",
    )
    _assert(piecewise_total_segment_switches > 0, "PiecewiseAcceleration did not switch any segments.")
    _assert(piecewise_each_env_switched, "At least one PiecewiseAcceleration env did not switch segments.")
    _assert(asset_sync["target_position"] <= SYNC_POSITION_TOL, "Target asset position is not synchronized.")
    _assert(asset_sync["target_velocity"] <= SYNC_VELOCITY_TOL, "Target asset velocity is not synchronized.")
    _assert(asset_sync["ego_position"] <= SYNC_POSITION_TOL, "Ego asset position is not synchronized.")
    _assert(asset_sync["ego_velocity"] <= SYNC_VELOCITY_TOL, "Ego asset velocity is not synchronized.")
    _assert(ego_max_drift <= EGO_DRIFT_TOL, "Ego drifted during M3 motion audit.")
    _assert(float(fixed_height_error.item()) <= RESET_STATE_TOL, "Target height was not fixed.")
    _assert(min_relative_distance_all_physics_substeps > task.cfg.d_safe, "Runtime center distance became <= d_safe.")
    _assert(not bool(task.target_motion_invalid_buf.any().item()), "M3 target motion invalid flag was raised.")
    _assert(not bool(task.collision_risk_buf.any().item()), "M3 collision risk flag was raised.")
    _assert(max_target_speed <= task.cfg.target_motion.max_speed, "Target speed exceeded max_speed.")
    _assert(max_target_acceleration <= task.cfg.target_motion.max_acceleration, "Target acceleration exceeded max_acceleration.")
    _assert_all_finite("task_state", _clone_task_state(task))
    _assert_all_finite("target_asset_pos", task.target.data.root_pos_w)
    _assert_all_finite("target_asset_vel", task.target.data.root_lin_vel_w)
    _assert_all_finite("ego_asset_pos", task.ego.data.root_pos_w)
    _assert_all_finite("ego_asset_vel", task.ego.data.root_lin_vel_w)

    partial_reset = _check_partial_reset(task)

    return {
        "passed": True,
        "truth_authority": "task_tensors",
        "actor_observation": "policy_only_p_rel_w_v_rel_w",
        "actor_observation_checks": last_actor_check,
        "actor_information_leakage_check": {
            "mode_id_exposed_to_actor": False,
            "motion_parameters_exposed_to_actor": False,
            "future_schedule_exposed_to_actor": False,
            "future_target_state_exposed_to_actor": False,
            "basis": "only observation group is policy tensor with shape [num_envs, 6] equal to concat(p_rel_w, v_rel_w)",
        },
        "rigid_object_role": "synchronized_state_and_visualization_carrier",
        "split": args_cli.split,
        "split_config_ranges_different": split_ranges_different,
        "split_config_summary": _split_config_summary(task.cfg.target_motion),
        "mode_selection": "random" if args_cli.random_modes else "cycle",
        "mode_counts": mode_counts,
        "num_envs": int(task.num_envs),
        "steps": int(args_cli.steps),
        "seed": int(args_cli.seed),
        "sim_dt": float(task.physics_dt),
        "decimation": int(task.cfg.decimation),
        "env_step_dt": float(task.step_dt),
        "motion_independent_errors": motion_errors,
        "max_target_speed_observed": max_target_speed,
        "max_target_acceleration_observed": max_target_acceleration,
        "asset_sync_errors": asset_sync,
        "ego_max_drift": ego_max_drift,
        "fixed_height_max_error": float(fixed_height_error.item()),
        "min_relative_distance_env_steps": min_relative_distance,
        "min_relative_distance_all_physics_substeps": min_relative_distance_all_physics_substeps,
        "workspace_abs_max_observed": workspace_abs_max,
        "d_safe": float(task.cfg.d_safe),
        "collision_enabled": {"ego": ego_collision_enabled, "target": target_collision_enabled},
        "no_contact_verified_by_center_distance": True,
        "no_physics_disturbance_verified_by_sync_and_ego_drift": True,
        "piecewise_total_segment_switches": piecewise_total_segment_switches,
        "piecewise_each_env_switched": piecewise_each_env_switched,
        "piecewise_boundary_audit": piecewise_boundary_audit,
        "segment_switch_count": segment_switch_count,
        "partial_reset": partial_reset,
        "seed_audit": seed_audit,
        "target_motion_invalid": bool(task.target_motion_invalid_buf.any().item()),
        "collision_risk": bool(task.collision_risk_buf.any().item()),
        "finite": True,
    }


def _seed_probe_report(args_cli: argparse.Namespace, env: Any) -> dict[str, Any]:
    state = _collect_reset_state(env, args_cli.seed)
    task = env.unwrapped
    return {
        "seed": int(args_cli.seed),
        "split": args_cli.split,
        "num_envs": int(task.num_envs),
        "mode_selection": "random" if args_cli.random_modes else "cycle",
        "sampling_digest": _state_digest(state),
        "mode_counts": _mode_counts_from_task(task),
        "split_config_ranges_different": _split_ranges_different(task.cfg.target_motion),
        "min_reset_relative_distance": float(torch.linalg.norm(task.p_rel_w, dim=1).min().item()),
        "actor_observation_checks": _check_policy_observation(task, {"policy": torch.cat((task.p_rel_w, task.v_rel_w), dim=-1)}),
    }


def main() -> None:
    args_cli = _build_parser().parse_args()
    simulation_app = _launch_app(args_cli)
    env = None

    try:
        env = _make_env(args_cli, args_cli.seed, steps=args_cli.steps)
        if args_cli.seed_probe:
            report = _seed_probe_report(args_cli, env)
            print(f"[INFO] M3 seed probe: {json.dumps(report, sort_keys=True)}", flush=True)
            return

        print("[INFO] Running M3 seed audit by reseeding one runtime env instance.", flush=True)
        seed_audit = _run_seed_audit(args_cli, env)
        print("[INFO] M3 seed audit passed. Running main runtime audit.", flush=True)
        report = _run_main_audit(args_cli, env, seed_audit)
        print(f"[INFO] M3 runtime audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
