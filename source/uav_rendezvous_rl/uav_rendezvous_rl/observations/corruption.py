"""Stateless vectorized random corruption helpers for M7A observations."""

from __future__ import annotations

import math

import torch


def _unit_interval(values: torch.Tensor) -> torch.Tensor:
    return values - torch.floor(values)


def stateless_uniform(
    env_ids: torch.Tensor,
    counters: torch.Tensor,
    dim: int,
    seed: int,
    stream: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic uniform samples in `(0, 1)` from seed, env id, and counter."""

    device = env_ids.device
    env = env_ids.to(dtype=torch.float32).reshape(-1, 1)
    step = counters.to(device=device, dtype=torch.float32).reshape(-1, 1)
    comp = torch.arange(int(dim), dtype=torch.float32, device=device).reshape(1, -1)
    phase = (
        (env + 1.0) * 12.9898
        + (step + 1.0) * 78.233
        + (comp + 1.0) * 37.719
        + float(int(seed) + 1) * 0.12345
        + float(int(stream) + 1) * 19.1919
    )
    values = _unit_interval(torch.sin(phase) * 43758.5453123)
    return torch.clamp(values.to(dtype=dtype), min=1.0e-6, max=1.0 - 1.0e-6)


def stateless_dropout_mask(env_ids: torch.Tensor, counters: torch.Tensor, probability: float, seed: int, stream: int) -> torch.Tensor:
    """Return one dropout decision per environment and observation step."""

    prob = float(probability)
    if prob <= 0.0:
        return torch.zeros(env_ids.shape[0], dtype=torch.bool, device=env_ids.device)
    if prob >= 1.0:
        return torch.ones(env_ids.shape[0], dtype=torch.bool, device=env_ids.device)
    samples = stateless_uniform(env_ids, counters, 1, seed, stream).squeeze(-1)
    return samples < prob


def stateless_normal(
    env_ids: torch.Tensor,
    counters: torch.Tensor,
    dim: int,
    seed: int,
    stream: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic standard normal samples using a Box-Muller transform."""

    u1 = stateless_uniform(env_ids, counters, dim, seed, stream, dtype=dtype)
    u2 = stateless_uniform(env_ids, counters, dim, seed, stream + 1, dtype=dtype)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
