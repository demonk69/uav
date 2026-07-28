"""M7B wind and gust process tests."""

import torch

from uav_rendezvous_rl.dynamics import WindGustState, sample_gust_segment, sample_steady_wind, step_wind_and_gust


def test_steady_wind_is_horizontal_and_bounded() -> None:
    env_ids = torch.arange(32, dtype=torch.long)
    episodes = torch.zeros(32, dtype=torch.long)

    wind = sample_steady_wind(env_ids, episodes, base_seed=42, max_magnitude=0.20, a_max_nominal=2.0, device=torch.device("cpu"))

    assert torch.allclose(wind[:, 2], torch.zeros(32))
    assert float(torch.linalg.norm(wind, dim=1).max().item()) <= 0.40 + 1e-6


def test_gust_segment_counter_changes_target_and_duration() -> None:
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    episodes = torch.zeros(3, dtype=torch.long)

    target_a, duration_a = sample_gust_segment(
        env_ids, episodes, 42, 0.15, 10, 50, 2.0, torch.device("cpu"), sample_counter=torch.zeros(3, dtype=torch.long)
    )
    target_b, duration_b = sample_gust_segment(
        env_ids, episodes, 42, 0.15, 10, 50, 2.0, torch.device("cpu"), sample_counter=torch.ones(3, dtype=torch.long)
    )

    assert not torch.allclose(target_a, target_b)
    assert int(duration_a.min().item()) >= 10
    assert int(duration_a.max().item()) <= 50
    assert int(duration_b.min().item()) >= 10
    assert int(duration_b.max().item()) <= 50


def test_gust_lowpass_updates_every_call_and_decrements_once() -> None:
    state = WindGustState(
        steady_wind_w=torch.zeros((1, 3)),
        gust_target_w=torch.tensor([[1.0, 0.0, 0.0]]),
        gust_current_w=torch.zeros((1, 3)),
        gust_remaining_policy_steps=torch.tensor([3], dtype=torch.long),
        gust_segment_index=torch.zeros(1, dtype=torch.long),
    )

    first = step_wind_and_gust(state, physics_dt=0.01, gust_tau_s=0.20, is_policy_step=True)
    remaining_after_first = int(state.gust_remaining_policy_steps[0].item())
    second = step_wind_and_gust(state, physics_dt=0.01, gust_tau_s=0.20, is_policy_step=False)

    assert remaining_after_first == 2
    assert int(state.gust_remaining_policy_steps[0].item()) == 2
    assert float(second[0, 0].item()) > float(first[0, 0].item())


def test_gust_reset_initialization_starts_at_target() -> None:
    target = torch.tensor([[0.2, 0.1, 0.0]])
    state = WindGustState(
        steady_wind_w=torch.zeros((1, 3)),
        gust_target_w=target.clone(),
        gust_current_w=target.clone(),
        gust_remaining_policy_steps=torch.tensor([10], dtype=torch.long),
        gust_segment_index=torch.zeros(1, dtype=torch.long),
    )

    wind = step_wind_and_gust(state, physics_dt=0.01, gust_tau_s=0.20, is_policy_step=True)
    assert torch.allclose(wind, target)
