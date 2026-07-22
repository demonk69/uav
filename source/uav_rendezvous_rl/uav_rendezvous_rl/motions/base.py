"""Common M3 target motion interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class MotionState:
    """Current target motion truth state in env-local world frame w."""

    p_target_w: torch.Tensor
    v_target_w: torch.Tensor
    a_target_w: torch.Tensor


class TargetMotionGenerator(Protocol):
    """Protocol for vectorized target motion managers/generators."""

    def reset(
        self,
        env_ids: torch.Tensor,
        initial_position_w: torch.Tensor,
        initial_velocity_w: torch.Tensor,
        generator: torch.Generator,
        split: str,
    ) -> MotionState:
        """Reset selected environments."""

    def evaluate(self, physics_step_count: torch.Tensor | None = None) -> MotionState:
        """Evaluate target state from integer physics step counts."""
