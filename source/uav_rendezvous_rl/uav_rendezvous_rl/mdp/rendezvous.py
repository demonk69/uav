"""Pure PyTorch helpers for the M5 feedforward rendezvous MDP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uav_rendezvous_rl.motions.configs import (
    MODE_CONSTANT_ACCELERATION,
    MODE_CONSTANT_TURN,
    MODE_CONSTANT_VELOCITY,
    MODE_PIECEWISE_ACCELERATION,
)


@dataclass(frozen=True)
class RendezvousActionCfg:
    """M5 action mapping and simplified ego velocity-tracking limits."""

    v_max: float = 3.0
    v_abs_max: float = 5.0
    a_max: float = 2.0
    tau_v: float = 0.25


@dataclass(frozen=True)
class RendezvousInitialGeometryCfg:
    """M5 randomized desired-offset and ego initial-geometry ranges."""

    delta_radial_range: tuple[float, float] = (2.0, 6.0)
    delta_tangent_range: tuple[float, float] = (-2.0, 2.0)
    min_initial_offset_error: float = 1.0
    max_resample_attempts: int = 8


@dataclass(frozen=True)
class RendezvousRewardCfg:
    """M5 reward scales and success thresholds."""

    sigma_offset: float = 1.5
    safety_buffer: float = 1.0
    offset_scale: float = 2.0
    relative_velocity_scale: float = 0.15
    progress_scale: float = 6.0
    action_smoothness_scale: float = 0.05
    action_magnitude_scale: float = 0.002
    safety_distance_scale: float = 2.0
    collision_penalty: float = 10.0
    speed_limit_scale: float = 0.05
    accel_limit_scale: float = 0.02
    attitude_rate_scale: float = 0.01
    workspace_penalty: float = 2.0
    success_step_bonus: float = 0.5
    success_completion_bonus: float = 5.0
    success_offset_error: float = 0.50
    success_relative_speed: float = 0.30
    success_hold_s: float = 1.0
    terminate_on_success: bool = False


@dataclass(frozen=True)
class RendezvousInitialGeometry:
    """Sampled M5 initial geometry tensors."""

    p_ego_w: torch.Tensor
    v_ego_w: torch.Tensor
    b_des_w: torch.Tensor
    theta: torch.Tensor
    delta_radial: torch.Tensor
    delta_tangent: torch.Tensor


def map_raw_action_to_velocity_command(raw_action: torch.Tensor, v_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unbounded Actor output to a world-frame velocity command."""

    squashed_action = torch.tanh(raw_action)
    return squashed_action, squashed_action * float(v_max)


def raw_action_from_velocity_command(v_cmd_w: torch.Tensor, v_max: float, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the raw action whose tanh maps to the requested velocity command."""

    normalized = torch.clamp(v_cmd_w / float(v_max), min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))


def identity_rotation_6d(num_envs: int, device: torch.device | str) -> torch.Tensor:
    """Return the fixed 6D identity-attitude representation used by M5 placeholders."""

    return torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=torch.float32, device=device).repeat(num_envs, 1)


def _uniform_range(
    value_range: tuple[float, float],
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    values = torch.rand(shape, generator=generator, dtype=torch.float32)
    low, high = value_range
    return (float(low) + (float(high) - float(low)) * values).to(device=device)


def _sample_desired_offset_w(
    num_envs: int,
    d_offset: float,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = _uniform_range((-torch.pi, torch.pi), (num_envs,), generator, device)
    b_des_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    b_des_w[:, 0] = float(d_offset) * torch.cos(theta)
    b_des_w[:, 1] = float(d_offset) * torch.sin(theta)
    return b_des_w, theta


def validate_m5_initial_geometry(
    p_target_initial_w: torch.Tensor,
    p_ego_initial_w: torch.Tensor,
    b_des_w: torch.Tensor,
    d_safe: float,
    min_initial_offset_error: float,
) -> dict[str, torch.Tensor]:
    """Validate M5 non-contact initial geometry for each environment."""

    offset_error_w = p_ego_initial_w - p_target_initial_w - b_des_w
    offset_error_norm = torch.linalg.norm(offset_error_w, dim=-1)
    center_distance = torch.linalg.norm(p_target_initial_w - p_ego_initial_w, dim=-1)
    b_norm = torch.linalg.norm(b_des_w, dim=-1)
    finite = torch.isfinite(p_target_initial_w).all(dim=-1)
    finite &= torch.isfinite(p_ego_initial_w).all(dim=-1)
    finite &= torch.isfinite(b_des_w).all(dim=-1)
    valid = finite
    valid &= center_distance > float(d_safe)
    valid &= offset_error_norm >= float(min_initial_offset_error)
    valid &= b_norm > float(d_safe)
    return {
        "valid": valid,
        "offset_error_norm": offset_error_norm,
        "center_distance": center_distance,
        "b_norm": b_norm,
    }


def sample_m5_initial_geometry(
    p_target_initial_w: torch.Tensor,
    d_offset: float,
    d_safe: float,
    cfg: RendezvousInitialGeometryCfg,
    generator: torch.Generator,
    device: torch.device | str,
) -> RendezvousInitialGeometry:
    """Sample M5 desired offset and ego initial state with bounded batch resampling."""

    num_envs = int(p_target_initial_w.shape[0])
    p_ego_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    v_ego_w = torch.zeros_like(p_ego_w)
    b_des_w = torch.zeros_like(p_ego_w)
    theta = torch.zeros(num_envs, dtype=torch.float32, device=device)
    delta_radial = torch.zeros(num_envs, dtype=torch.float32, device=device)
    delta_tangent = torch.zeros(num_envs, dtype=torch.float32, device=device)
    valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for _ in range(int(cfg.max_resample_attempts)):
        resample_ids = torch.nonzero(~valid_mask, as_tuple=False).squeeze(-1)
        if resample_ids.numel() == 0:
            break
        count = int(resample_ids.numel())
        b_candidate_w, theta_candidate = _sample_desired_offset_w(count, d_offset, generator, device)
        radial = b_candidate_w / float(d_offset)
        tangent = torch.stack((-radial[:, 1], radial[:, 0], torch.zeros(count, dtype=torch.float32, device=device)), dim=-1)
        delta_r = _uniform_range(cfg.delta_radial_range, (count,), generator, device)
        delta_t = _uniform_range(cfg.delta_tangent_range, (count,), generator, device)
        p_candidate_w = p_target_initial_w[resample_ids] + b_candidate_w
        p_candidate_w = p_candidate_w + delta_r.unsqueeze(-1) * radial + delta_t.unsqueeze(-1) * tangent

        geometry = validate_m5_initial_geometry(
            p_target_initial_w[resample_ids],
            p_candidate_w,
            b_candidate_w,
            d_safe,
            cfg.min_initial_offset_error,
        )
        accepted = geometry["valid"]
        if torch.any(accepted):
            accepted_ids = resample_ids[accepted]
            p_ego_w[accepted_ids] = p_candidate_w[accepted]
            b_des_w[accepted_ids] = b_candidate_w[accepted]
            theta[accepted_ids] = theta_candidate[accepted]
            delta_radial[accepted_ids] = delta_r[accepted]
            delta_tangent[accepted_ids] = delta_t[accepted]
            valid_mask[accepted_ids] = True

    if not bool(torch.all(valid_mask).item()):
        raise RuntimeError("M5 initial geometry sampling failed to satisfy safety constraints within max attempts.")
    return RendezvousInitialGeometry(p_ego_w, v_ego_w, b_des_w, theta, delta_radial, delta_tangent)


def assemble_actor_observation(
    p_rel_w: torch.Tensor,
    v_rel_w: torch.Tensor,
    v_ego_w: torch.Tensor,
    r_ego_6d: torch.Tensor,
    omega_ego_b: torch.Tensor,
    previous_squashed_action: torch.Tensor,
    b_des_w: torch.Tensor,
    d_offset: float | torch.Tensor,
) -> torch.Tensor:
    """Assemble the fixed 25D M5 deployable Actor observation."""

    if isinstance(d_offset, torch.Tensor):
        d_offset_column = d_offset.to(device=p_rel_w.device, dtype=torch.float32).reshape(-1, 1)
    else:
        d_offset_column = torch.full((p_rel_w.shape[0], 1), float(d_offset), dtype=torch.float32, device=p_rel_w.device)
    return torch.cat(
        (
            p_rel_w,
            v_rel_w,
            v_ego_w,
            r_ego_6d,
            omega_ego_b,
            previous_squashed_action,
            b_des_w,
            d_offset_column,
        ),
        dim=-1,
    )


def encode_target_motion_current_params(
    mode_id: torch.Tensor,
    v0_w: torch.Tensor,
    constant_acceleration_w: torch.Tensor,
    turn_omega: torch.Tensor,
    current_acceleration_w: torch.Tensor,
    max_speed: float,
    max_acceleration: float,
    max_turn_omega: float,
) -> torch.Tensor:
    """Encode current target-generator parameters as a padded normalized 6D critic-only vector."""

    params = torch.zeros((mode_id.shape[0], 6), dtype=torch.float32, device=mode_id.device)
    speed_scale = max(float(max_speed), 1.0e-6)
    acceleration_scale = max(float(max_acceleration), 1.0e-6)
    turn_scale = max(float(max_turn_omega), 1.0e-6)

    cv_mask = mode_id == MODE_CONSTANT_VELOCITY
    ca_mask = mode_id == MODE_CONSTANT_ACCELERATION
    ct_mask = mode_id == MODE_CONSTANT_TURN
    pw_mask = mode_id == MODE_PIECEWISE_ACCELERATION

    params[cv_mask, 0:3] = v0_w[cv_mask] / speed_scale
    params[ca_mask, 0:3] = constant_acceleration_w[ca_mask] / acceleration_scale
    params[ct_mask, 0:2] = v0_w[ct_mask, 0:2] / speed_scale
    params[ct_mask, 2] = turn_omega[ct_mask] / turn_scale
    params[pw_mask, 0:3] = current_acceleration_w[pw_mask] / acceleration_scale
    return torch.clamp(params, -10.0, 10.0)


def assemble_critic_observation(
    actor_obs: torch.Tensor,
    p_ego_w: torch.Tensor,
    p_target_w: torch.Tensor,
    v_target_w: torch.Tensor,
    a_target_w: torch.Tensor,
    r_target_6d: torch.Tensor,
    omega_target_b: torch.Tensor,
    mode_one_hot: torch.Tensor,
    target_motion_current_params: torch.Tensor,
    episode_phase: torch.Tensor,
) -> torch.Tensor:
    """Assemble the fixed 57D M5 privileged Critic observation."""

    return torch.cat(
        (
            actor_obs,
            p_ego_w,
            p_target_w,
            v_target_w,
            a_target_w,
            r_target_6d,
            omega_target_b,
            mode_one_hot.to(dtype=torch.float32),
            target_motion_current_params,
            episode_phase.reshape(-1, 1).to(dtype=torch.float32),
        ),
        dim=-1,
    )


def compute_reward_terms(
    offset_error_w: torch.Tensor,
    previous_offset_error_norm: torch.Tensor,
    v_ego_w: torch.Tensor,
    v_target_w: torch.Tensor,
    raw_action: torch.Tensor,
    action_delta_squashed: torch.Tensor,
    center_distance: torch.Tensor,
    acceleration_saturated: torch.Tensor,
    speed_saturated: torch.Tensor,
    collision_risk: torch.Tensor,
    workspace_violation: torch.Tensor,
    success_step: torch.Tensor,
    success_completed: torch.Tensor,
    d_safe: float,
    omega_ego_b: torch.Tensor,
    cfg: RendezvousRewardCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Compute M5 reward and separately logged component tensors."""

    offset_error_norm = torch.linalg.norm(offset_error_w, dim=-1)
    relative_speed = torch.linalg.norm(v_ego_w - v_target_w, dim=-1)
    safe_margin = center_distance - float(d_safe)
    safety_violation = torch.relu(float(cfg.safety_buffer) - safe_margin)

    terms = {
        "offset": float(cfg.offset_scale) * torch.exp(-(offset_error_norm.square()) / float(cfg.sigma_offset) ** 2),
        "relative_velocity": -float(cfg.relative_velocity_scale) * relative_speed.square(),
        "progress": float(cfg.progress_scale) * (previous_offset_error_norm - offset_error_norm),
        "action_smoothness": -float(cfg.action_smoothness_scale) * torch.sum(action_delta_squashed.square(), dim=-1),
        "action_magnitude": -float(cfg.action_magnitude_scale) * torch.sum(raw_action.square(), dim=-1),
        "safety_distance": -float(cfg.safety_distance_scale) * safety_violation.square()
        - float(cfg.collision_penalty) * collision_risk.to(dtype=torch.float32),
        "speed_limit": -float(cfg.speed_limit_scale) * speed_saturated.to(dtype=torch.float32),
        "accel_limit": -float(cfg.accel_limit_scale) * acceleration_saturated.to(dtype=torch.float32),
        "attitude_rate": -float(cfg.attitude_rate_scale) * torch.sum(omega_ego_b.square(), dim=-1),
        "workspace": -float(cfg.workspace_penalty) * workspace_violation.to(dtype=torch.float32),
        "success_bonus": float(cfg.success_step_bonus) * success_step.to(dtype=torch.float32)
        + float(cfg.success_completion_bonus) * success_completed.to(dtype=torch.float32),
    }
    reward = torch.sum(torch.stack(tuple(terms.values()), dim=0), dim=0)
    return reward, terms, offset_error_norm
