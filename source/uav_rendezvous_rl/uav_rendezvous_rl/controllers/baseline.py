"""Pure PyTorch M4 deterministic offset rendezvous baseline."""

from __future__ import annotations

import torch

from .configs import BaselineInitialGeometryCfg


def clamp_vector_norm(vector_w: torch.Tensor, max_norm: float, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp batched vectors by Euclidean norm without changing direction."""

    norm = torch.linalg.norm(vector_w, dim=-1, keepdim=True)
    max_norm_tensor = torch.as_tensor(float(max_norm), dtype=vector_w.dtype, device=vector_w.device)
    scale = torch.clamp(max_norm_tensor / torch.clamp(norm, min=eps), max=1.0)
    saturated = norm.squeeze(-1) > float(max_norm)
    return vector_w * scale, saturated


def compute_target_prediction(p_target_w: torch.Tensor, v_target_w: torch.Tensor, prediction_horizon_s: float) -> torch.Tensor:
    """Predict target position with current-state constant-velocity extrapolation."""

    return p_target_w + v_target_w * float(prediction_horizon_s)


def compute_baseline_velocity_command(
    p_ego_w: torch.Tensor,
    p_target_w: torch.Tensor,
    v_target_w: torch.Tensor,
    b_des_w: torch.Tensor,
    prediction_horizon_s: float,
    v_cmd_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute M4 offset-goal velocity command from current deployable state only.

    The command is `(p_target_w + v_target_w*T + b_des_w - p_ego_w) / T`,
    equivalently `v_target_w + (p_target_w + b_des_w - p_ego_w) / T`.
    """

    horizon = float(prediction_horizon_s)
    p_target_pred_w = compute_target_prediction(p_target_w, v_target_w, horizon)
    p_goal_w = p_target_pred_w + b_des_w
    raw_v_cmd_w = (p_goal_w - p_ego_w) / horizon
    v_cmd_w, velocity_saturated = clamp_vector_norm(raw_v_cmd_w, v_cmd_max)
    return v_cmd_w, velocity_saturated, p_target_pred_w, p_goal_w


def compute_limited_acceleration(
    v_cmd_w: torch.Tensor,
    v_ego_w: torch.Tensor,
    tau_v: float,
    a_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track the velocity command with acceleration norm limiting."""

    raw_a_cmd_w = (v_cmd_w - v_ego_w) / float(tau_v)
    return clamp_vector_norm(raw_a_cmd_w, a_max)


def integrate_ego_kinematics(
    p_ego_w: torch.Tensor,
    v_ego_w: torch.Tensor,
    a_cmd_w: torch.Tensor,
    physics_dt: float,
    v_abs_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate acceleration-limited ego kinematics for one physics substep."""

    dt = float(physics_dt)
    p_next_w = p_ego_w + v_ego_w * dt + 0.5 * a_cmd_w * dt * dt
    raw_v_next_w = v_ego_w + a_cmd_w * dt
    v_next_w, speed_saturated = clamp_vector_norm(raw_v_next_w, v_abs_max)
    return p_next_w, v_next_w, speed_saturated


def line_segment_point_distance(segment_start_w: torch.Tensor, segment_end_w: torch.Tensor, point_w: torch.Tensor) -> torch.Tensor:
    """Return minimum distance from a point to each batched line segment."""

    segment_w = segment_end_w - segment_start_w
    point_rel_w = point_w - segment_start_w
    denom = torch.sum(segment_w * segment_w, dim=-1).clamp_min(1.0e-12)
    u = torch.sum(point_rel_w * segment_w, dim=-1) / denom
    u = torch.clamp(u, 0.0, 1.0).unsqueeze(-1)
    closest_w = segment_start_w + u * segment_w
    return torch.linalg.norm(point_w - closest_w, dim=-1)


def sample_baseline_initial_ego_state(
    p_target_initial_w: torch.Tensor,
    b_des_w: torch.Tensor,
    cfg: BaselineInitialGeometryCfg,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ego outside the desired offset point, after target initialization."""

    num_envs = p_target_initial_w.shape[0]
    delta_initial_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    delta_x = torch.rand((num_envs,), generator=generator, dtype=torch.float32)
    delta_y = torch.rand((num_envs,), generator=generator, dtype=torch.float32)
    dx_low, dx_high = cfg.delta_x_range
    dy_low, dy_high = cfg.delta_y_range
    delta_initial_w[:, 0] = (dx_low + (dx_high - dx_low) * delta_x).to(device=device)
    delta_initial_w[:, 1] = (dy_low + (dy_high - dy_low) * delta_y).to(device=device)
    delta_initial_w[:, 2] = float(cfg.delta_z)
    p_ego_initial_w = p_target_initial_w + b_des_w + delta_initial_w
    v_ego_initial_w = torch.zeros_like(p_ego_initial_w)
    return p_ego_initial_w, v_ego_initial_w, delta_initial_w


def validate_baseline_initial_geometry(
    p_target_initial_w: torch.Tensor,
    p_ego_initial_w: torch.Tensor,
    b_des_w: torch.Tensor,
    d_safe: float,
    min_initial_offset_error: float,
) -> dict[str, torch.Tensor]:
    """Validate that initial approach geometry is non-contact and outside the offset point."""

    p_goal_initial_w = p_target_initial_w + b_des_w
    offset_error_norm = torch.linalg.norm(p_ego_initial_w - p_goal_initial_w, dim=-1)
    center_distance = torch.linalg.norm(p_target_initial_w - p_ego_initial_w, dim=-1)
    segment_distance = line_segment_point_distance(p_ego_initial_w, p_goal_initial_w, p_target_initial_w)
    valid = center_distance > float(d_safe)
    valid &= offset_error_norm >= float(min_initial_offset_error)
    valid &= segment_distance > float(d_safe)
    valid &= torch.isfinite(p_ego_initial_w).all(dim=-1)
    return {
        "valid": valid,
        "offset_error_norm": offset_error_norm,
        "center_distance": center_distance,
        "segment_distance": segment_distance,
    }
