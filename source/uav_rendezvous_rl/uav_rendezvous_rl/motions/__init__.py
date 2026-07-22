"""Vectorized target motion library for UAV rendezvous milestones."""

from .base import MotionState, TargetMotionGenerator
from .configs import (
    MODE_CONSTANT_ACCELERATION,
    MODE_CONSTANT_TURN,
    MODE_CONSTANT_VELOCITY,
    MODE_NAMES,
    MODE_PIECEWISE_ACCELERATION,
    TargetMotionManagerCfg,
    TargetMotionSplitCfg,
)
from .manager import TargetMotionManager

__all__ = [
    "MODE_CONSTANT_ACCELERATION",
    "MODE_CONSTANT_TURN",
    "MODE_CONSTANT_VELOCITY",
    "MODE_NAMES",
    "MODE_PIECEWISE_ACCELERATION",
    "MotionState",
    "TargetMotionGenerator",
    "TargetMotionManager",
    "TargetMotionManagerCfg",
    "TargetMotionSplitCfg",
]
