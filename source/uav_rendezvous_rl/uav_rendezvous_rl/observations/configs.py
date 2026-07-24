"""Configuration for M7A causal observation degradation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationPipelineCfg:
    """Immutable runtime-independent configuration for the M7A observation pipeline."""

    position_delay_steps: int = 0
    velocity_delay_steps: int = 0
    position_update_period_steps: int = 1
    velocity_update_period_steps: int = 1
    position_dropout_prob: float = 0.0
    velocity_dropout_prob: float = 0.0
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    seed_offset: int = 7301

    @property
    def history_length(self) -> int:
        return max(int(self.position_delay_steps), int(self.velocity_delay_steps)) + 1

    def validate(self) -> None:
        if int(self.position_delay_steps) < 0 or int(self.velocity_delay_steps) < 0:
            raise ValueError("Observation delay steps must be non-negative.")
        if int(self.position_update_period_steps) < 1 or int(self.velocity_update_period_steps) < 1:
            raise ValueError("Observation update periods must be positive integers.")
        if not 0.0 <= float(self.position_dropout_prob) <= 1.0:
            raise ValueError("Position dropout probability must be in [0, 1].")
        if not 0.0 <= float(self.velocity_dropout_prob) <= 1.0:
            raise ValueError("Velocity dropout probability must be in [0, 1].")
        if float(self.position_noise_std) < 0.0 or float(self.velocity_noise_std) < 0.0:
            raise ValueError("Observation noise standard deviations must be non-negative.")


_STAGE_ALIASES = {
    "0": 0,
    "stage0": 0,
    "clean": 0,
    "1": 1,
    "stage1": 1,
    "velocity_low_freq": 1,
    "velocity-low-freq": 1,
    "2": 2,
    "stage2": 2,
    "medium_delay": 2,
    "medium-delay": 2,
    "3": 3,
    "stage3": 3,
    "dropout": 3,
    "4": 4,
    "stage4": 4,
    "combined": 4,
}


def _normalize_stage(stage: int | str) -> int:
    if isinstance(stage, int):
        stage_id = stage
    else:
        key = str(stage).strip().lower()
        if key not in _STAGE_ALIASES:
            raise ValueError(f"Unknown M7A observation stage: {stage!r}.")
        stage_id = _STAGE_ALIASES[key]
    if stage_id not in (0, 1, 2, 3, 4):
        raise ValueError(f"Unknown M7A observation stage: {stage!r}.")
    return stage_id


def make_m7a_observation_cfg(stage: int | str) -> ObservationPipelineCfg:
    """Return the fixed M7A observation-degradation config for a stage."""

    stage_id = _normalize_stage(stage)
    if stage_id == 0:
        return ObservationPipelineCfg()
    if stage_id == 1:
        return ObservationPipelineCfg(velocity_update_period_steps=5)
    if stage_id == 2:
        return ObservationPipelineCfg(position_delay_steps=1, velocity_delay_steps=3)
    if stage_id == 3:
        return ObservationPipelineCfg(position_dropout_prob=0.05, velocity_dropout_prob=0.10)
    return ObservationPipelineCfg(
        position_delay_steps=1,
        velocity_delay_steps=3,
        velocity_update_period_steps=5,
        position_dropout_prob=0.05,
        velocity_dropout_prob=0.10,
        position_noise_std=0.02,
        velocity_noise_std=0.02,
    )
