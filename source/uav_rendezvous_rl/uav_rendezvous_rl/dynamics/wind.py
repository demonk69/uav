"""M7B steady wind and piecewise gust model with physics-substep low-pass smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .stateless_rng import stateless_m7b_angle, stateless_m7b_randint, stateless_m7b_uniform


@dataclass
class WindGustState:
    """Per-environment wind and gust runtime state tensors."""

    steady_wind_w: torch.Tensor
    gust_target_w: torch.Tensor
    gust_current_w: torch.Tensor
    gust_remaining_policy_steps: torch.Tensor
    gust_segment_index: torch.Tensor


def _make_wind_direction(
    env_ids: torch.Tensor,
    episode_counts: torch.Tensor,
    base_seed: int,
    angle_stream: int,
    sample_counter: int | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    theta = stateless_m7b_angle(env_ids, episode_counts, base_seed, angle_stream, sample_counter)
    direction = torch.zeros((env_ids.numel(), 3), dtype=dtype, device=device)
    direction[:, 0] = torch.cos(theta)
    direction[:, 1] = torch.sin(theta)
    return direction


def sample_steady_wind(
    env_ids: torch.Tensor,
    episode_counts: torch.Tensor,
    base_seed: int,
    max_magnitude: float,
    a_max_nominal: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    max_accel = float(max_magnitude) * float(a_max_nominal)
    magnitude = stateless_m7b_uniform(env_ids, episode_counts, base_seed, 1010, 0, dtype).squeeze(-1) * max_accel
    direction = _make_wind_direction(env_ids, episode_counts, base_seed, 1011, 0, device, dtype)
    return direction * magnitude.unsqueeze(-1)


def sample_gust_segment(
    env_ids: torch.Tensor,
    episode_counts: torch.Tensor,
    base_seed: int,
    max_magnitude: float,
    min_duration: int,
    max_duration: int,
    a_max_nominal: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    sample_counter: int | torch.Tensor = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_accel = float(max_magnitude) * float(a_max_nominal)
    magnitude = stateless_m7b_uniform(env_ids, episode_counts, base_seed, 1021, sample_counter, dtype).squeeze(-1) * max_accel
    direction = _make_wind_direction(env_ids, episode_counts, base_seed, 1022, sample_counter, device, dtype)
    duration = stateless_m7b_randint(
        env_ids, episode_counts, int(min_duration), int(max_duration), base_seed, 1020, sample_counter
    )
    return direction * magnitude.unsqueeze(-1), duration


def step_wind_and_gust(
    state: WindGustState,
    physics_dt: float,
    gust_tau_s: float,
    is_policy_step: bool,
) -> torch.Tensor:
    """Update gust state and return current wind acceleration.

    Every call advances gust current toward gust target via low-pass filter.
    Set ``is_policy_step`` true exactly once per policy step to decrement the
    policy-step segment counter.
    """

    alpha = 1.0 - torch.exp(torch.as_tensor(-float(physics_dt) / float(gust_tau_s)))
    state.gust_current_w[:] = state.gust_current_w + alpha * (state.gust_target_w - state.gust_current_w)

    if is_policy_step:
        state.gust_remaining_policy_steps[:] = state.gust_remaining_policy_steps - 1

    return state.steady_wind_w + state.gust_current_w
