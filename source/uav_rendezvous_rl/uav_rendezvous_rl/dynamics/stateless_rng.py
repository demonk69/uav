"""Stateless counter-based deterministic RNG for M7B dynamics randomization."""

from __future__ import annotations

import torch


def _unit_interval(values: torch.Tensor) -> torch.Tensor:
    return values - torch.floor(values)


def stateless_m7b_uniform(
    env_ids: torch.Tensor,
    episode_counts: torch.Tensor,
    base_seed: int,
    stream: int,
    sample_counter: int | torch.Tensor = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic uniform samples in (0, 1) keyed by (env_id, episode_count, stream, counter)."""

    device = env_ids.device
    env = env_ids.to(dtype=torch.float32).reshape(-1, 1)
    episode = episode_counts.to(device=device, dtype=torch.float32).reshape(-1, 1)
    if isinstance(sample_counter, torch.Tensor):
        counter = sample_counter.to(device=device, dtype=torch.float32).reshape(-1, 1)
    else:
        counter = torch.full_like(env, float(int(sample_counter)))
    phase = (
        (env + 1.0) * 12.9898
        + (episode + 1.0) * 78.233
        + float(int(base_seed) + 1) * 0.12345
        + float(int(stream) + 1) * 19.1919
        + (counter + 1.0) * 41.777
    )
    values = _unit_interval(torch.sin(phase) * 43758.5453123)
    return torch.clamp(values.to(dtype=dtype), min=1.0e-6, max=1.0 - 1.0e-6)


def stateless_m7b_angle(
    env_ids: torch.Tensor,
    episode_counts: torch.Tensor,
    base_seed: int,
    stream: int,
    sample_counter: int | torch.Tensor = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic angles in [0, 2*pi)."""

    uniform = stateless_m7b_uniform(env_ids, episode_counts, base_seed, stream, sample_counter, dtype)
    return uniform.squeeze(-1) * (2.0 * 3.141592653589793)


def stateless_m7b_randint(
    env_ids: torch.Tensor,
    episode_counts: torch.Tensor,
    low: int,
    high: int,
    base_seed: int,
    stream: int,
    sample_counter: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Return deterministic uniform integer samples in [low, high]."""

    uniform = stateless_m7b_uniform(env_ids, episode_counts, base_seed, stream, sample_counter)
    return (low + uniform.squeeze(-1) * float(high - low + 1)).to(dtype=torch.long).clamp(low, high)
