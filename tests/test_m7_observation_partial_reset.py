"""M7A observation pipeline partial-reset tests."""

import torch

from uav_rendezvous_rl.observations import ObservationPipeline, ObservationPipelineCfg


def _values(base: float, num_envs: int = 5) -> torch.Tensor:
    data = torch.zeros((num_envs, 3), dtype=torch.float32)
    data[:, 0] = torch.arange(num_envs, dtype=torch.float32) + base
    return data


def test_partial_reset_changes_only_selected_environments() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=2, velocity_delay_steps=2, velocity_update_period_steps=3)
    pipeline = ObservationPipeline(cfg, num_envs=5, device="cpu", seed=17)
    pipeline.reset(None, _values(0.0), _values(100.0))
    for step in range(4):
        pipeline.observe(_values(float(step)), _values(float(100 + step)))
    before = pipeline.runtime_state()

    reset_ids = torch.tensor([1, 3], dtype=torch.long)
    p_reset = torch.tensor([[1000.0, 0.0, 0.0], [3000.0, 0.0, 0.0]], dtype=torch.float32)
    v_reset = p_reset + 10.0
    pipeline.reset(reset_ids, p_reset, v_reset)
    after = pipeline.runtime_state()
    untouched = torch.tensor([0, 2, 4], dtype=torch.long)

    for key in before:
        assert torch.equal(after[key][untouched], before[key][untouched]), key
    assert torch.equal(after["position_history"][1], p_reset[0].repeat(cfg.history_length, 1))
    assert torch.equal(after["position_history"][3], p_reset[1].repeat(cfg.history_length, 1))
    assert torch.equal(after["velocity_history"][1], v_reset[0].repeat(cfg.history_length, 1))
    assert torch.equal(after["velocity_history"][3], v_reset[1].repeat(cfg.history_length, 1))
    assert torch.equal(after["step_count"][reset_ids], torch.zeros(2, dtype=torch.long))


def test_partial_reset_first_observation_uses_new_episode_initial_sample() -> None:
    cfg = ObservationPipelineCfg(position_delay_steps=2, velocity_delay_steps=2)
    pipeline = ObservationPipeline(cfg, num_envs=3, device="cpu", seed=17)
    pipeline.reset(None, _values(0.0, 3), _values(100.0, 3))
    for step in range(4):
        pipeline.observe(_values(float(step), 3), _values(float(100 + step), 3))

    pipeline.reset([1], torch.tensor([[99.0, 0.0, 0.0]]), torch.tensor([[199.0, 0.0, 0.0]]))
    p_truth = _values(10.0, 3)
    v_truth = _values(110.0, 3)
    p_truth[1, 0] = 100.0
    v_truth[1, 0] = 200.0
    p_obs, v_obs = pipeline.observe(p_truth, v_truth)

    assert float(p_obs[1, 0].item()) == 99.0
    assert float(v_obs[1, 0].item()) == 199.0
