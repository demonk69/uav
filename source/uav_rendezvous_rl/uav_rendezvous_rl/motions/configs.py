"""Configuration objects for M3 target motion."""

from __future__ import annotations

from dataclasses import dataclass, field


MODE_CONSTANT_VELOCITY = 0
MODE_CONSTANT_ACCELERATION = 1
MODE_CONSTANT_TURN = 2
MODE_PIECEWISE_ACCELERATION = 3

MODE_NAMES = {
    MODE_CONSTANT_VELOCITY: "ConstantVelocity",
    MODE_CONSTANT_ACCELERATION: "ConstantAcceleration",
    MODE_CONSTANT_TURN: "ConstantTurn",
    MODE_PIECEWISE_ACCELERATION: "PiecewiseAcceleration",
}


@dataclass(frozen=True)
class TargetMotionSplitCfg:
    """Split-specific sampling ranges and seed domain."""

    name: str
    seed_offset: int
    mode_probabilities: tuple[float, float, float, float]
    target_pos_x_range: tuple[float, float]
    target_pos_y_range: tuple[float, float]
    target_height_range: tuple[float, float]
    target_vel_x_range: tuple[float, float]
    target_vel_y_range: tuple[float, float]
    acceleration_x_range: tuple[float, float]
    acceleration_y_range: tuple[float, float]
    turn_omega_range: tuple[float, float]
    piecewise_acceleration_x_range: tuple[float, float]
    piecewise_acceleration_y_range: tuple[float, float]
    piecewise_segment_duration_steps_range: tuple[int, int]


@dataclass(frozen=True)
class TargetMotionManagerCfg:
    """Top-level target motion configuration."""

    fixed_height: bool = True
    d_safe: float = 0.75
    max_speed: float = 8.0
    max_acceleration: float = 0.25
    validation_horizon_s: float = 120.0
    omega_epsilon: float = 1.0e-5
    max_initial_resample_attempts: int = 8
    workspace_x_range: tuple[float, float] = (-250.0, 250.0)
    workspace_y_range: tuple[float, float] = (-250.0, 250.0)
    workspace_z_range: tuple[float, float] = (1.5, 1.5)
    force_mode_cycle_on_reset: bool = False
    train: TargetMotionSplitCfg = field(
        default_factory=lambda: TargetMotionSplitCfg(
            name="train",
            seed_offset=0,
            mode_probabilities=(0.25, 0.25, 0.25, 0.25),
            target_pos_x_range=(4.0, 8.0),
            target_pos_y_range=(-2.0, 2.0),
            target_height_range=(1.5, 1.5),
            target_vel_x_range=(0.2, 1.0),
            target_vel_y_range=(-0.5, 0.5),
            acceleration_x_range=(-0.015, 0.015),
            acceleration_y_range=(-0.015, 0.015),
            turn_omega_range=(-0.10, 0.10),
            piecewise_acceleration_x_range=(-0.02, 0.02),
            piecewise_acceleration_y_range=(-0.02, 0.02),
            piecewise_segment_duration_steps_range=(25, 120),
        )
    )
    validation: TargetMotionSplitCfg = field(
        default_factory=lambda: TargetMotionSplitCfg(
            name="validation",
            seed_offset=10_000,
            mode_probabilities=(0.25, 0.25, 0.25, 0.25),
            target_pos_x_range=(5.0, 9.0),
            target_pos_y_range=(-2.5, 2.5),
            target_height_range=(1.5, 1.5),
            target_vel_x_range=(0.25, 0.9),
            target_vel_y_range=(-0.6, 0.6),
            acceleration_x_range=(-0.012, 0.012),
            acceleration_y_range=(-0.012, 0.012),
            turn_omega_range=(-0.08, 0.08),
            piecewise_acceleration_x_range=(-0.018, 0.018),
            piecewise_acceleration_y_range=(-0.018, 0.018),
            piecewise_segment_duration_steps_range=(40, 160),
        )
    )
    test: TargetMotionSplitCfg = field(
        default_factory=lambda: TargetMotionSplitCfg(
            name="test",
            seed_offset=20_000,
            mode_probabilities=(0.25, 0.25, 0.25, 0.25),
            target_pos_x_range=(8.0, 12.0),
            target_pos_y_range=(-3.0, 3.0),
            target_height_range=(1.5, 1.5),
            target_vel_x_range=(0.3, 1.2),
            target_vel_y_range=(-0.75, 0.75),
            acceleration_x_range=(-0.010, 0.010),
            acceleration_y_range=(-0.010, 0.010),
            turn_omega_range=(-0.12, 0.12),
            piecewise_acceleration_x_range=(-0.015, 0.015),
            piecewise_acceleration_y_range=(-0.015, 0.015),
            piecewise_segment_duration_steps_range=(60, 200),
        )
    )


def get_split_cfg(cfg: TargetMotionManagerCfg, split: str) -> TargetMotionSplitCfg:
    """Return split config by name."""

    if split == "train":
        return cfg.train
    if split == "validation":
        return cfg.validation
    if split == "test":
        return cfg.test
    raise ValueError(f"Unknown target motion split: {split!r}.")


def validate_motion_config(cfg: TargetMotionManagerCfg) -> None:
    """Validate ranges that can be checked without sampling."""

    for split in (cfg.train, cfg.validation, cfg.test):
        if split.piecewise_segment_duration_steps_range[0] < 1:
            raise ValueError(f"{split.name} segment duration must be at least one physics step.")
        if len(split.mode_probabilities) != 4 or any(prob < 0.0 for prob in split.mode_probabilities):
            raise ValueError(f"{split.name} mode probabilities must be four non-negative values.")
        if sum(split.mode_probabilities) <= 0.0:
            raise ValueError(f"{split.name} mode probabilities must have positive sum.")
        max_v = max(abs(split.target_vel_x_range[0]), abs(split.target_vel_x_range[1]))
        max_v += max(abs(split.target_vel_y_range[0]), abs(split.target_vel_y_range[1]))
        max_a = max(abs(split.acceleration_x_range[0]), abs(split.acceleration_x_range[1]))
        max_a += max(abs(split.acceleration_y_range[0]), abs(split.acceleration_y_range[1]))
        if max_v + max_a * cfg.validation_horizon_s > cfg.max_speed:
            raise ValueError(f"{split.name} constant-acceleration range can exceed max_speed.")
        max_piecewise_a = max(
            abs(split.piecewise_acceleration_x_range[0]), abs(split.piecewise_acceleration_x_range[1])
        ) + max(abs(split.piecewise_acceleration_y_range[0]), abs(split.piecewise_acceleration_y_range[1]))
        if max_piecewise_a > cfg.max_acceleration:
            raise ValueError(f"{split.name} piecewise acceleration range exceeds max_acceleration.")
        max_turn_speed = (
            max(abs(split.target_vel_x_range[0]), abs(split.target_vel_x_range[1])) ** 2
            + max(abs(split.target_vel_y_range[0]), abs(split.target_vel_y_range[1])) ** 2
        ) ** 0.5
        max_turn_acceleration = max_turn_speed * max(abs(split.turn_omega_range[0]), abs(split.turn_omega_range[1]))
        if max_turn_acceleration > cfg.max_acceleration:
            raise ValueError(f"{split.name} constant-turn range exceeds max_acceleration.")
