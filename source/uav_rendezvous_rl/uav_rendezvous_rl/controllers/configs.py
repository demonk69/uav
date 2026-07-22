"""Configuration dataclasses for the M4 deterministic baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineControllerCfg:
    """M4 short-horizon offset rendezvous controller limits."""

    prediction_horizon_s: float = 0.5
    v_cmd_max: float = 3.0
    v_abs_max: float = 5.0
    a_max: float = 2.0
    tau_v: float = 0.25
    success_offset_error: float = 0.50
    success_relative_speed: float = 0.30
    success_hold_s: float = 1.0
    terminate_on_success: bool = False


@dataclass(frozen=True)
class BaselineInitialGeometryCfg:
    """M4 initial ego offset ranges relative to target plus desired offset."""

    delta_x_range: tuple[float, float] = (2.0, 6.0)
    delta_y_range: tuple[float, float] = (-2.0, 2.0)
    delta_z: float = 0.0
    min_initial_offset_error: float = 1.0
    max_resample_attempts: int = 8
