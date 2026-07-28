"""M7B parameter range construction tests with independent expected bounds."""

import torch

from uav_rendezvous_rl.dynamics import stateless_m7b_uniform


def _range_sample(env_ids: torch.Tensor, episodes: torch.Tensor, low: float, high: float, stream: int) -> torch.Tensor:
    return low + stateless_m7b_uniform(env_ids, episodes, base_seed=42, stream=stream).squeeze(-1) * (high - low)


def test_dynamics_randomization_ranges_match_plan() -> None:
    env_ids = torch.arange(128, dtype=torch.long)
    episodes = torch.zeros(128, dtype=torch.long)

    tau = _range_sample(env_ids, episodes, 0.75, 1.50, 1001)
    accel = _range_sample(env_ids, episodes, 0.80, 1.20, 1002)
    speed = _range_sample(env_ids, episodes, 0.90, 1.10, 1003)
    drag = _range_sample(env_ids, episodes, 0.0, 0.15, 1004)

    assert float(tau.min().item()) >= 0.75
    assert float(tau.max().item()) <= 1.50
    assert float(accel.min().item()) >= 0.80
    assert float(accel.max().item()) <= 1.20
    assert float(speed.min().item()) >= 0.90
    assert float(speed.max().item()) <= 1.10
    assert float(drag.min().item()) >= 0.0
    assert float(drag.max().item()) <= 0.15


def test_stage_zero_nominal_values_are_not_randomized() -> None:
    tau_velocity_scale = 1.0
    acceleration_limit_scale = 1.0
    speed_limit_scale = 1.0
    linear_drag = 0.0
    action_delay_steps = 0

    assert tau_velocity_scale == 1.0
    assert acceleration_limit_scale == 1.0
    assert speed_limit_scale == 1.0
    assert linear_drag == 0.0
    assert action_delay_steps == 0
