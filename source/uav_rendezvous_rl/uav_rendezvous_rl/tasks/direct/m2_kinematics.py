"""Pure PyTorch helpers for the M2 truth-state environment.

These helpers intentionally implement only constant-velocity, fixed-height target
motion for M2. They are not a general target motion generator library.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class M2RandomizationCfg:
    """Randomization ranges for M2 reset sampling."""

    ego_initial_pos_w: tuple[float, float, float]
    target_pos_x_range: tuple[float, float]
    target_pos_y_range: tuple[float, float]
    target_height_range: tuple[float, float]
    target_vel_x_range: tuple[float, float]
    target_vel_y_range: tuple[float, float]
    d_safe: float


def _uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    values = torch.rand(shape, generator=generator, dtype=torch.float32)
    return (low + (high - low) * values).to(device=device)


def sample_m2_initial_conditions(
    num_envs: int,
    cfg: M2RandomizationCfg,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample M2 ego/target initial truth states in each env-local world frame.

    Returns:
        Tuple of `(p_ego_w, v_ego_w, p_target_initial_w, v_target_w)`.
    """

    p_ego_w = torch.tensor(cfg.ego_initial_pos_w, dtype=torch.float32, device=device).repeat(num_envs, 1)
    v_ego_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    p_target_initial_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    p_target_initial_w[:, 0] = _uniform(*cfg.target_pos_x_range, (num_envs,), generator, device)
    p_target_initial_w[:, 1] = _uniform(*cfg.target_pos_y_range, (num_envs,), generator, device)
    p_target_initial_w[:, 2] = _uniform(*cfg.target_height_range, (num_envs,), generator, device)

    v_target_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    v_target_w[:, 0] = _uniform(*cfg.target_vel_x_range, (num_envs,), generator, device)
    v_target_w[:, 1] = _uniform(*cfg.target_vel_y_range, (num_envs,), generator, device)

    relative_distance = torch.linalg.norm(p_target_initial_w - p_ego_w, dim=1)
    if torch.any(relative_distance <= cfg.d_safe):
        raise RuntimeError("M2 reset sampling produced an initial center distance <= d_safe.")

    return p_ego_w, v_ego_w, p_target_initial_w, v_target_w


def compute_constant_velocity_position_w(
    p_initial_w: torch.Tensor,
    v_w: torch.Tensor,
    elapsed_time: torch.Tensor,
) -> torch.Tensor:
    """Compute fixed-height constant-velocity target position analytically."""

    p_w = p_initial_w + v_w * elapsed_time.unsqueeze(-1)
    p_w[:, 2] = p_initial_w[:, 2]
    return p_w


def compute_relative_state_w(
    p_ego_w: torch.Tensor,
    v_ego_w: torch.Tensor,
    p_target_w: torch.Tensor,
    v_target_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed project relative-state definitions in world-aligned frame w."""

    return p_target_w - p_ego_w, v_target_w - v_ego_w


def compute_offset_error_w(p_ego_w: torch.Tensor, p_target_w: torch.Tensor, b_des_w: torch.Tensor) -> torch.Tensor:
    """Compute fixed project offset-error definition in world-aligned frame w."""

    return p_ego_w - p_target_w - b_des_w


def all_finite(*tensors: torch.Tensor) -> bool:
    """Return True if all tensors contain only finite values."""

    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
