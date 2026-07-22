"""Vectorized piecewise constant-acceleration target motion."""

from __future__ import annotations

import torch

from .base import MotionState
from .constant_acceleration import compute_constant_acceleration


def compute_piecewise_acceleration(
    segment_start_position_w: torch.Tensor,
    segment_start_velocity_w: torch.Tensor,
    current_acceleration_w: torch.Tensor,
    segment_step_count: torch.Tensor,
    physics_dt: float,
    fixed_height: bool = True,
) -> MotionState:
    """Evaluate current piecewise segment from integer segment step counts."""

    return compute_constant_acceleration(
        segment_start_position_w,
        segment_start_velocity_w,
        current_acceleration_w,
        segment_step_count,
        physics_dt,
        fixed_height=fixed_height,
    )
