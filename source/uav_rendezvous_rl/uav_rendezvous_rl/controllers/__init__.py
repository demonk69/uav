"""Deterministic controllers for UAV rendezvous milestones."""

from .baseline import (
    clamp_vector_norm,
    compute_baseline_velocity_command,
    compute_limited_acceleration,
    compute_target_prediction,
    integrate_ego_kinematics,
    line_segment_point_distance,
    sample_baseline_initial_ego_state,
    validate_baseline_initial_geometry,
)
from .configs import BaselineControllerCfg, BaselineInitialGeometryCfg

__all__ = [
    "BaselineControllerCfg",
    "BaselineInitialGeometryCfg",
    "clamp_vector_norm",
    "compute_baseline_velocity_command",
    "compute_limited_acceleration",
    "compute_target_prediction",
    "integrate_ego_kinematics",
    "line_segment_point_distance",
    "sample_baseline_initial_ego_state",
    "validate_baseline_initial_geometry",
]
