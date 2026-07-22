"""Vectorized fixed-height constant-velocity target motion."""

from __future__ import annotations

import torch

from .base import MotionState


def compute_constant_velocity(
    p0_w: torch.Tensor,
    v0_w: torch.Tensor,
    step_count: torch.Tensor,
    physics_dt: float,
    fixed_height: bool = True,
) -> MotionState:
    """Evaluate `p(t)=p0+v0*t`, `v(t)=v0`, `a(t)=0` from integer step counts."""

    dtype = p0_w.dtype
    t = step_count.to(device=p0_w.device, dtype=torch.float64).unsqueeze(-1) * float(physics_dt)
    p_w = p0_w.to(torch.float64) + v0_w.to(torch.float64) * t
    v_w = v0_w.to(torch.float64).clone()
    a_w = torch.zeros_like(v_w)
    if fixed_height:
        p_w[:, 2] = p0_w[:, 2].to(torch.float64)
        v_w[:, 2] = 0.0
    return MotionState(p_w.to(dtype), v_w.to(dtype), a_w.to(dtype))
