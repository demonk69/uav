"""M7A low-frequency update and sample-and-hold tests."""

import torch

from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg


def _value(value: float) -> torch.Tensor:
    return torch.tensor([[value, value + 1.0, value + 2.0]], dtype=torch.float32)


def test_velocity_sample_and_hold_update_period_is_integer_counted() -> None:
    cfg = ObservationPipelineCfg(velocity_update_period_steps=5)
    pipeline = ObservationPipeline(cfg, num_envs=1, device="cpu", seed=11)
    pipeline.reset(None, _value(0.0), _value(100.0))

    velocity_x = []
    for step in range(7):
        _, v_obs = pipeline.observe(_value(float(step)), _value(float(100 + step)))
        velocity_x.append(float(v_obs[0, 0].item()))

    assert velocity_x == [100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 105.0]


def test_position_stays_current_when_update_period_is_one() -> None:
    cfg = ObservationPipelineCfg(position_update_period_steps=1, velocity_update_period_steps=5)
    pipeline = ObservationPipeline(cfg, num_envs=1, device="cpu", seed=11)
    pipeline.reset(None, _value(0.0), _value(100.0))

    position_x = []
    for step in range(4):
        p_obs, _ = pipeline.observe(_value(float(step)), _value(float(100 + step)))
        position_x.append(float(p_obs[0, 0].item()))

    assert position_x == [0.0, 1.0, 2.0, 3.0]


def test_sample_hold_uses_delayed_sample_when_both_are_enabled() -> None:
    cfg = ObservationPipelineCfg(velocity_delay_steps=2, velocity_update_period_steps=3)
    pipeline = ObservationPipeline(cfg, num_envs=1, device="cpu", seed=11)
    pipeline.reset(None, _value(0.0), _value(100.0))

    velocity_x = []
    for step in range(7):
        _, v_obs = pipeline.observe(_value(float(step)), _value(float(100 + step)))
        velocity_x.append(float(v_obs[0, 0].item()))

    assert velocity_x == [100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 104.0]
