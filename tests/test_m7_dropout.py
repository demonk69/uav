"""M7A dropout behavior."""

import torch

from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg


def _sample(value: float, num_envs: int = 2) -> torch.Tensor:
    tensor = torch.zeros((num_envs, 3), dtype=torch.float32)
    tensor[:, 0] = value
    return tensor


def test_dropout_holds_last_valid_value_without_exposing_mask() -> None:
    cfg = ObservationPipelineCfg(position_dropout_prob=1.0, velocity_dropout_prob=1.0)
    pipeline = ObservationPipeline(cfg, num_envs=2, device="cpu", seed=3)
    pipeline.reset(None, _sample(10.0), _sample(20.0))

    p_obs, v_obs = pipeline.observe(_sample(11.0), _sample(21.0))
    p_next, v_next = pipeline.observe(_sample(12.0), _sample(22.0))

    assert torch.equal(p_obs, _sample(10.0))
    assert torch.equal(v_obs, _sample(20.0))
    assert torch.equal(p_next, _sample(10.0))
    assert torch.equal(v_next, _sample(20.0))
    assert torch.all(pipeline.last_position_dropout_mask)
    assert torch.all(pipeline.last_velocity_dropout_mask)


def test_no_dropout_updates_last_valid_value() -> None:
    pipeline = ObservationPipeline(ObservationPipelineCfg(), num_envs=2, device="cpu", seed=3)
    pipeline.reset(None, _sample(10.0), _sample(20.0))

    p_obs, v_obs = pipeline.observe(_sample(11.0), _sample(21.0))

    pipeline.observe(_sample(12.0), _sample(22.0))

    assert torch.equal(p_obs, _sample(11.0))
    assert torch.equal(v_obs, _sample(21.0))
    assert torch.equal(pipeline.last_valid_position, _sample(12.0))
    assert torch.equal(pipeline.last_valid_velocity, _sample(22.0))


def test_dropout_seed_reproducibility_and_seed_difference() -> None:
    cfg = ObservationPipelineCfg(position_dropout_prob=0.5, velocity_dropout_prob=0.5)
    first = ObservationPipeline(cfg, num_envs=64, device="cpu", seed=9)
    second = ObservationPipeline(cfg, num_envs=64, device="cpu", seed=9)
    third = ObservationPipeline(cfg, num_envs=64, device="cpu", seed=10)
    zero = torch.zeros((64, 3), dtype=torch.float32)
    one = torch.ones((64, 3), dtype=torch.float32)
    for pipeline in (first, second, third):
        pipeline.reset(None, zero, zero)
        pipeline.observe(one, one)

    assert torch.equal(first.last_position_dropout_mask, second.last_position_dropout_mask)
    assert torch.equal(first.last_velocity_dropout_mask, second.last_velocity_dropout_mask)
    assert not torch.equal(first.last_position_dropout_mask, third.last_position_dropout_mask)
