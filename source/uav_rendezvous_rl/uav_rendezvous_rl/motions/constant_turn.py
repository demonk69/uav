"""Vectorized fixed-height constant-turn target motion."""

from __future__ import annotations

import torch

from .base import MotionState


def compute_constant_turn(
    p0_w: torch.Tensor,
    v0_w: torch.Tensor,
    omega: torch.Tensor,
    step_count: torch.Tensor,
    physics_dt: float,
    fixed_height: bool = True,
    omega_epsilon: float = 1.0e-5,
) -> MotionState:
    """Evaluate horizontal coordinated turn with closed-form position integration."""

    dtype = p0_w.dtype
    device = p0_w.device
    t = step_count.to(device=device, dtype=torch.float64) * float(physics_dt)
    omega64 = omega.to(device=device, dtype=torch.float64)
    theta = omega64 * t
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    vx0 = v0_w[:, 0].to(torch.float64)
    vy0 = v0_w[:, 1].to(torch.float64)

    v_x = vx0 * cos_theta - vy0 * sin_theta
    v_y = vx0 * sin_theta + vy0 * cos_theta
    safe_omega = torch.where(torch.abs(omega64) < omega_epsilon, torch.ones_like(omega64), omega64)
    dx_turn = (vx0 * sin_theta + vy0 * (cos_theta - 1.0)) / safe_omega
    dy_turn = (vx0 * (1.0 - cos_theta) + vy0 * sin_theta) / safe_omega
    dx_cv = vx0 * t
    dy_cv = vy0 * t
    small_omega = torch.abs(omega64) < omega_epsilon
    dx = torch.where(small_omega, dx_cv, dx_turn)
    dy = torch.where(small_omega, dy_cv, dy_turn)
    v_x = torch.where(small_omega, vx0, v_x)
    v_y = torch.where(small_omega, vy0, v_y)

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
    return MotionState(p_w.to(dtype), v_w.to(dtype), a_w.to(dtype))
