"""M7A observation noise seed and statistics tests."""

import torch

from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg


def test_noise_is_seed_reproducible_and_does_not_change_truth() -> None:
    cfg = ObservationPipelineCfg(position_noise_std=0.05, velocity_noise_std=0.07)
    first = ObservationPipeline(cfg, num_envs=128, device="cpu", seed=123)
    second = ObservationPipeline(cfg, num_envs=128, device="cpu", seed=123)
    truth_p = torch.full((128, 3), 2.0)
    truth_v = torch.full((128, 3), -1.0)
    for pipeline in (first, second):
        pipeline.reset(None, truth_p, truth_v)

    p_first, v_first = first.observe(truth_p, truth_v)
    p_second, v_second = second.observe(truth_p, truth_v)

    assert torch.equal(p_first, p_second)
    assert torch.equal(v_first, v_second)
    assert torch.equal(truth_p, torch.full((128, 3), 2.0))
    assert torch.equal(truth_v, torch.full((128, 3), -1.0))


def test_different_noise_seeds_produce_different_samples() -> None:
    cfg = ObservationPipelineCfg(position_noise_std=0.05, velocity_noise_std=0.05)
    first = ObservationPipeline(cfg, num_envs=128, device="cpu", seed=123)
    second = ObservationPipeline(cfg, num_envs=128, device="cpu", seed=124)
    truth = torch.zeros((128, 3), dtype=torch.float32)
    for pipeline in (first, second):
        pipeline.reset(None, truth, truth)
    p_first, v_first = first.observe(truth, truth)
    p_second, v_second = second.observe(truth, truth)

    assert not torch.equal(p_first, p_second)
    assert not torch.equal(v_first, v_second)


def test_noise_is_approximately_zero_mean_with_configured_scale() -> None:
    cfg = ObservationPipelineCfg(position_noise_std=0.10, velocity_noise_std=0.20)
    pipeline = ObservationPipeline(cfg, num_envs=20000, device="cpu", seed=321)
    truth = torch.zeros((20000, 3), dtype=torch.float32)
    pipeline.reset(None, truth, truth)

    p_obs, v_obs = pipeline.observe(truth, truth)

    assert abs(float(p_obs.mean().item())) < 0.01
    assert abs(float(v_obs.mean().item())) < 0.02
    assert 0.08 < float(p_obs.std(unbiased=False).item()) < 0.12
    assert 0.16 < float(v_obs.std(unbiased=False).item()) < 0.24
