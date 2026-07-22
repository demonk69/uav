"""Pure tensor accounting helpers for M4 runtime diagnostics."""

from __future__ import annotations

import torch


def update_collision_risk_accounting(
    collision: torch.Tensor,
    collision_risk_buf: torch.Tensor,
    collision_risk_count: torch.Tensor,
) -> torch.Tensor:
    """Mark first collision-risk detection per episode and return newly detected envs."""

    new_collision = collision & ~collision_risk_buf
    collision_risk_buf[:] |= collision
    collision_risk_count += new_collision.to(dtype=collision_risk_count.dtype)
    return new_collision
