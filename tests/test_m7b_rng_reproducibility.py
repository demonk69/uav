"""M7B stateless RNG reproducibility and reset-order invariance tests."""

import torch

from uav_rendezvous_rl.dynamics import stateless_m7b_randint, stateless_m7b_uniform


def test_stateless_uniform_is_env_order_invariant() -> None:
    env_a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    eps_a = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    env_b = torch.tensor([3, 1, 0, 2], dtype=torch.long)
    eps_b = torch.tensor([8, 6, 5, 7], dtype=torch.long)

    samples_a = stateless_m7b_uniform(env_a, eps_a, base_seed=42, stream=1001).squeeze(-1)
    samples_b = stateless_m7b_uniform(env_b, eps_b, base_seed=42, stream=1001).squeeze(-1)

    by_env_a = {int(env.item()): float(value.item()) for env, value in zip(env_a, samples_a, strict=True)}
    by_env_b = {int(env.item()): float(value.item()) for env, value in zip(env_b, samples_b, strict=True)}
    assert by_env_a == by_env_b


def test_stateless_uniform_is_independent_across_streams() -> None:
    env_ids = torch.arange(8, dtype=torch.long)
    episodes = torch.ones(8, dtype=torch.long)

    tau = stateless_m7b_uniform(env_ids, episodes, base_seed=42, stream=1001)
    accel = stateless_m7b_uniform(env_ids, episodes, base_seed=42, stream=1002)

    assert not torch.allclose(tau, accel)


def test_stateless_gust_segment_counter_changes_samples() -> None:
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    episodes = torch.tensor([3, 3, 3], dtype=torch.long)

    first = stateless_m7b_uniform(env_ids, episodes, base_seed=42, stream=1021, sample_counter=torch.zeros(3, dtype=torch.long))
    second = stateless_m7b_uniform(env_ids, episodes, base_seed=42, stream=1021, sample_counter=torch.ones(3, dtype=torch.long))

    assert not torch.allclose(first, second)


def test_stateless_randint_bounds_are_inclusive() -> None:
    env_ids = torch.arange(64, dtype=torch.long)
    episodes = torch.arange(64, dtype=torch.long)

    values = stateless_m7b_randint(env_ids, episodes, low=0, high=3, base_seed=7, stream=1005)

    assert values.dtype == torch.long
    assert int(values.min().item()) >= 0
    assert int(values.max().item()) <= 3
