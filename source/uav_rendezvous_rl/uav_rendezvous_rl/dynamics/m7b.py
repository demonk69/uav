"""M7B formal dynamics integration and config validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class M7BDynamicsConfig:
    tau_v_nominal: float = 0.25
    a_max_nominal: float = 2.0
    v_abs_max_nominal: float = 5.0
    v_max: float = 3.0
    a_applied_abs_max: float = 4.0
    max_delay_steps: int = 3
    gust_tau_s: float = 0.20
    gust_duration_min: int = 10
    gust_duration_max: int = 50


def validate_m7b_dynamics_config(cfg: M7BDynamicsConfig) -> None:
    for name, value in (
        ("tau_v_nominal", cfg.tau_v_nominal),
        ("a_max_nominal", cfg.a_max_nominal),
        ("v_abs_max_nominal", cfg.v_abs_max_nominal),
        ("v_max", cfg.v_max),
        ("a_applied_abs_max", cfg.a_applied_abs_max),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if int(cfg.max_delay_steps) < 0:
        raise ValueError(f"max_delay_steps must be non-negative, got {cfg.max_delay_steps}.")
    if float(cfg.gust_tau_s) <= 0.0:
        raise ValueError(f"gust_tau_s must be positive, got {cfg.gust_tau_s}.")


def _clamp_vector_norm_per_env(vector_w: torch.Tensor, max_norm: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp batched vectors by per-environment Euclidean norm limits."""

    norm = torch.linalg.norm(vector_w, dim=-1, keepdim=True)
    limit = max_norm.reshape(-1, 1).to(dtype=vector_w.dtype, device=vector_w.device)
    scale = torch.clamp(limit / torch.clamp(norm, min=eps), max=1.0)
    saturated = norm.squeeze(-1) > limit.squeeze(-1)
    return vector_w * scale, saturated


def integrate_m7b_dynamics(
    p_ego_w: torch.Tensor,
    v_ego_w: torch.Tensor,
    v_cmd_delayed_w: torch.Tensor,
    wind_acceleration_w: torch.Tensor,
    physics_dt: float,
    tau_velocity: torch.Tensor,
    acceleration_limit: torch.Tensor,
    physical_speed_limit: torch.Tensor,
    linear_drag: torch.Tensor,
    a_applied_abs_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance ego position and velocity by one physics substep.

    Returns:
        p_ego_next_w, v_ego_next_w, tracking_saturated, total_safety_saturated, speed_saturated
    """

    dt = float(physics_dt)
    tau = tau_velocity.reshape(-1, 1).to(dtype=p_ego_w.dtype, device=p_ego_w.device)
    a_lim = acceleration_limit.reshape(-1).to(dtype=p_ego_w.dtype, device=p_ego_w.device)
    speed_lim = physical_speed_limit.reshape(-1).to(dtype=p_ego_w.dtype, device=p_ego_w.device)
    drag = linear_drag.reshape(-1, 1).to(dtype=p_ego_w.dtype, device=p_ego_w.device)

    a_track_raw_w = (v_cmd_delayed_w - v_ego_w) / tau
    a_track_w, tracking_saturated = _clamp_vector_norm_per_env(a_track_raw_w, a_lim)

    a_unclipped_w = a_track_w - drag * v_ego_w + wind_acceleration_w
    a_applied_w, total_safety_saturated = _clamp_vector_norm_per_env(a_unclipped_w, torch.full_like(a_lim, float(a_applied_abs_max)))

    v_candidate_w = v_ego_w + a_applied_w * dt
    v_ego_next_w, speed_saturated = _clamp_vector_norm_per_env(v_candidate_w, speed_lim)

    p_ego_next_w = p_ego_w + v_ego_w * dt + 0.5 * a_applied_w * dt * dt

    return p_ego_next_w, v_ego_next_w, tracking_saturated, total_safety_saturated, speed_saturated
