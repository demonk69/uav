"""Sampling helpers for M3 target motion."""

from __future__ import annotations

import torch

from .configs import TargetMotionManagerCfg, TargetMotionSplitCfg, get_split_cfg


def make_split_generator(base_seed: int, cfg: TargetMotionManagerCfg, split: str) -> torch.Generator:
    """Create a CPU generator in the split-specific seed domain."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(base_seed) + int(get_split_cfg(cfg, split).seed_offset))
    return generator


def uniform_range(
    value_range: tuple[float, float],
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    values = torch.rand(shape, generator=generator, dtype=torch.float32)
    low, high = value_range
    return (low + (high - low) * values).to(device=device)


def randint_range(
    value_range: tuple[int, int],
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    low, high = value_range
    return torch.randint(low, high + 1, shape, generator=generator, dtype=torch.long).to(device=device)


def sample_mode_ids(
    num_envs: int,
    env_ids: torch.Tensor,
    split_cfg: TargetMotionSplitCfg,
    generator: torch.Generator,
    device: torch.device | str,
    force_cycle: bool,
) -> torch.Tensor:
    """Sample or cycle per-env mode ids."""

    if force_cycle:
        return torch.remainder(env_ids.to(device=device, dtype=torch.long), 4)
    probabilities = torch.tensor(split_cfg.mode_probabilities, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum()
    return torch.multinomial(probabilities, num_envs, replacement=True, generator=generator).to(device=device)


def sample_initial_target_state(
    num_envs: int,
    p_ego_w: torch.Tensor,
    cfg: TargetMotionManagerCfg,
    split_cfg: TargetMotionSplitCfg,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample safe target initial positions and velocities with bounded batch resampling."""

    p_target_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    v_target_w = torch.zeros_like(p_target_w)
    safe_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for _ in range(cfg.max_initial_resample_attempts):
        resample_ids = torch.nonzero(~safe_mask, as_tuple=False).squeeze(-1)
        if resample_ids.numel() == 0:
            break
        count = int(resample_ids.numel())
        p_candidate = torch.zeros((count, 3), dtype=torch.float32, device=device)
        p_candidate[:, 0] = uniform_range(split_cfg.target_pos_x_range, (count,), generator, device)
        p_candidate[:, 1] = uniform_range(split_cfg.target_pos_y_range, (count,), generator, device)
        p_candidate[:, 2] = uniform_range(split_cfg.target_height_range, (count,), generator, device)
        distance = torch.linalg.norm(p_candidate - p_ego_w[resample_ids], dim=1)
        candidate_safe = distance > cfg.d_safe
        if torch.any(candidate_safe):
            accepted_ids = resample_ids[candidate_safe]
            p_target_w[accepted_ids] = p_candidate[candidate_safe]
            safe_mask[accepted_ids] = True
    if not bool(torch.all(safe_mask).item()):
        raise RuntimeError("M3 initial target sampling failed to satisfy d_safe within max attempts.")

    v_target_w[:, 0] = uniform_range(split_cfg.target_vel_x_range, (num_envs,), generator, device)
    v_target_w[:, 1] = uniform_range(split_cfg.target_vel_y_range, (num_envs,), generator, device)
    return p_target_w, v_target_w


def sample_acceleration(
    num_envs: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    acceleration_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    acceleration_w[:, 0] = uniform_range(x_range, (num_envs,), generator, device)
    acceleration_w[:, 1] = uniform_range(y_range, (num_envs,), generator, device)
    return acceleration_w
