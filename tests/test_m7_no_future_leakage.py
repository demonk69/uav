"""M7A causal pulse and ramp tests."""

import torch

from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg


def _sample(value: float) -> torch.Tensor:
    return torch.tensor([[value, 0.0, 0.0]], dtype=torch.float32)


def test_pulse_input_does_not_appear_before_configured_delay() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=2, velocity_delay_steps=2)
    pipeline = ObservationPipeline(cfg, num_envs=1, device="cpu", seed=5)
    pipeline.reset(None, _sample(0.0), _sample(0.0))

    observed = []
    for step in range(8):
        value = 10.0 if step == 4 else 0.0
        p_obs, _ = pipeline.observe(_sample(value), _sample(value))
        observed.append(float(p_obs[0, 0].item()))

    assert observed == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]


def test_ramp_input_has_no_off_by_one_error() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=1, velocity_delay_steps=3)
    pipeline = ObservationPipeline(cfg, num_envs=1, device="cpu", seed=5)
    pipeline.reset(None, _sample(0.0), _sample(100.0))

    p_values = []
    v_values = []
    for step in range(6):
        p_obs, v_obs = pipeline.observe(_sample(float(step)), _sample(float(100 + step)))
        p_values.append(float(p_obs[0, 0].item()))
        v_values.append(float(v_obs[0, 0].item()))

    assert p_values == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    assert v_values == [100.0, 100.0, 100.0, 100.0, 101.0, 102.0]
