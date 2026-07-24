"""M7A observation delay semantics."""

import torch

from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg


def _column(value: float, num_envs: int = 2) -> torch.Tensor:
    out = torch.zeros((num_envs, 3), dtype=torch.float32)
    out[:, 0] = value
    return out


def test_delay_zero_returns_current_sample() -> None:
    pipeline = ObservationPipeline(ObservationPipelineCfg(), num_envs=2, device="cpu", seed=7)
    pipeline.reset(None, _column(0.0), _column(100.0))

    p_obs, v_obs = pipeline.observe(_column(3.0), _column(103.0))

    assert torch.equal(p_obs, _column(3.0))
    assert torch.equal(v_obs, _column(103.0))


def test_delay_one_returns_previous_sample() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=1, velocity_delay_steps=1)
    pipeline = ObservationPipeline(cfg, num_envs=2, device="cpu", seed=7)
    pipeline.reset(None, _column(0.0), _column(100.0))

    p_first, v_first = pipeline.observe(_column(1.0), _column(101.0))
    p_second, v_second = pipeline.observe(_column(2.0), _column(102.0))

    assert torch.equal(p_first, _column(0.0))
    assert torch.equal(v_first, _column(100.0))
    assert torch.equal(p_second, _column(1.0))
    assert torch.equal(v_second, _column(101.0))


def test_multi_step_delay_index_is_correct() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=3, velocity_delay_steps=3)
    pipeline = ObservationPipeline(cfg, num_envs=2, device="cpu", seed=7)
    pipeline.reset(None, _column(0.0), _column(100.0))

    observed = []
    for step in range(6):
        p_obs, _ = pipeline.observe(_column(float(step)), _column(float(100 + step)))
        observed.append(float(p_obs[0, 0].item()))

    assert observed == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]


def test_startup_buffer_is_initialized_with_reset_sample() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=4, velocity_delay_steps=2)
    pipeline = ObservationPipeline(cfg, num_envs=3, device="cpu", seed=7)
    p_initial = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    v_initial = p_initial + 10.0

    pipeline.reset(None, p_initial, v_initial)

    assert torch.equal(pipeline.position_history.data, p_initial.unsqueeze(1).expand(-1, cfg.history_length, -1))
    assert torch.equal(pipeline.velocity_history.data, v_initial.unsqueeze(1).expand(-1, cfg.history_length, -1))


def test_reset_first_frame_has_no_previous_episode_leakage() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=3, velocity_delay_steps=3)
    pipeline = ObservationPipeline(cfg, num_envs=1, device="cpu", seed=7)
    pipeline.reset(None, _column(50.0, 1), _column(150.0, 1))
    for step in range(5):
        pipeline.observe(_column(float(50 + step), 1), _column(float(150 + step), 1))

    pipeline.reset(None, _column(-2.0, 1), _column(-5.0, 1))
    p_obs, v_obs = pipeline.observe(_column(-1.0, 1), _column(-4.0, 1))

    assert torch.equal(p_obs, _column(-2.0, 1))
    assert torch.equal(v_obs, _column(-5.0, 1))
