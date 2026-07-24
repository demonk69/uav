"""Strictly causal M7A observation-degradation pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import copy

import torch

from .configs import ObservationPipelineCfg
from .corruption import stateless_dropout_mask, stateless_normal
from .history_buffer import ObservationHistoryBuffer


class ObservationPipeline:
    """Causal relative-state observation pipeline for vectorized environments."""

    def __init__(
        self,
        cfg: ObservationPipelineCfg,
        num_envs: int,
        device: torch.device | str,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ):
        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = int(seed) + int(cfg.seed_offset)
        self.env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.position_history = ObservationHistoryBuffer(self.num_envs, 3, cfg.history_length, self.device, dtype)
        self.velocity_history = ObservationHistoryBuffer(self.num_envs, 3, cfg.history_length, self.device, dtype)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.held_position = torch.zeros((self.num_envs, 3), dtype=dtype, device=self.device)
        self.held_velocity = torch.zeros_like(self.held_position)
        self.last_valid_position = torch.zeros_like(self.held_position)
        self.last_valid_velocity = torch.zeros_like(self.held_position)
        self.last_position_obs = torch.zeros_like(self.held_position)
        self.last_velocity_obs = torch.zeros_like(self.held_position)
        self.last_position_truth = torch.zeros_like(self.held_position)
        self.last_velocity_truth = torch.zeros_like(self.held_position)
        self.last_delayed_position = torch.zeros_like(self.held_position)
        self.last_delayed_velocity = torch.zeros_like(self.held_position)
        self.last_position_update_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_velocity_update_mask = torch.zeros_like(self.last_position_update_mask)
        self.last_position_dropout_mask = torch.zeros_like(self.last_position_update_mask)
        self.last_velocity_dropout_mask = torch.zeros_like(self.last_position_update_mask)
        self.last_position_read_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_velocity_read_age_steps = torch.zeros_like(self.last_position_read_age_steps)
        self.observation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.position_update_count = torch.zeros_like(self.observation_count)
        self.velocity_update_count = torch.zeros_like(self.observation_count)
        self.position_dropout_count = torch.zeros_like(self.observation_count)
        self.velocity_dropout_count = torch.zeros_like(self.observation_count)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return self.env_ids
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        p_rel_initial_w: torch.Tensor,
        v_rel_initial_w: torch.Tensor,
    ) -> None:
        """Reset only selected environments with current initial measurements."""

        ids = self._resolve_env_ids(env_ids)
        position = p_rel_initial_w.to(device=self.device, dtype=self.dtype)
        velocity = v_rel_initial_w.to(device=self.device, dtype=self.dtype)
        self.position_history.reset(ids, position)
        self.velocity_history.reset(ids, velocity)
        self.step_count[ids] = 0
        self.held_position[ids] = position
        self.held_velocity[ids] = velocity
        self.last_valid_position[ids] = position
        self.last_valid_velocity[ids] = velocity
        self.last_position_obs[ids] = position
        self.last_velocity_obs[ids] = velocity
        self.last_position_truth[ids] = position
        self.last_velocity_truth[ids] = velocity
        self.last_delayed_position[ids] = position
        self.last_delayed_velocity[ids] = velocity
        self.last_position_update_mask[ids] = True
        self.last_velocity_update_mask[ids] = True
        self.last_position_dropout_mask[ids] = False
        self.last_velocity_dropout_mask[ids] = False
        self.last_position_read_age_steps[ids] = 0
        self.last_velocity_read_age_steps[ids] = 0
        self.observation_count[ids] = 0
        self.position_update_count[ids] = 0
        self.velocity_update_count[ids] = 0
        self.position_dropout_count[ids] = 0
        self.velocity_dropout_count[ids] = 0

    def observe(self, p_rel_truth_w: torch.Tensor, v_rel_truth_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose only current or past relative-state samples after causal degradation."""

        position_truth = p_rel_truth_w.to(device=self.device, dtype=self.dtype)
        velocity_truth = v_rel_truth_w.to(device=self.device, dtype=self.dtype)
        self.last_position_truth[:] = position_truth
        self.last_velocity_truth[:] = velocity_truth

        self.position_history.push(position_truth)
        self.velocity_history.push(velocity_truth)
        delayed_position = self.position_history.read(int(self.cfg.position_delay_steps)).clone()
        delayed_velocity = self.velocity_history.read(int(self.cfg.velocity_delay_steps)).clone()
        self.last_delayed_position[:] = delayed_position
        self.last_delayed_velocity[:] = delayed_velocity

        position_update = (self.step_count % int(self.cfg.position_update_period_steps)) == 0
        velocity_update = (self.step_count % int(self.cfg.velocity_update_period_steps)) == 0
        self.held_position[:] = torch.where(position_update.unsqueeze(-1), delayed_position, self.held_position)
        self.held_velocity[:] = torch.where(velocity_update.unsqueeze(-1), delayed_velocity, self.held_velocity)

        position_dropout = stateless_dropout_mask(self.env_ids, self.step_count, self.cfg.position_dropout_prob, self.seed, 11)
        velocity_dropout = stateless_dropout_mask(self.env_ids, self.step_count, self.cfg.velocity_dropout_prob, self.seed, 23)
        valid_position = torch.where(position_dropout.unsqueeze(-1), self.last_valid_position, self.held_position)
        valid_velocity = torch.where(velocity_dropout.unsqueeze(-1), self.last_valid_velocity, self.held_velocity)
        self.last_valid_position[:] = torch.where(position_dropout.unsqueeze(-1), self.last_valid_position, self.held_position)
        self.last_valid_velocity[:] = torch.where(velocity_dropout.unsqueeze(-1), self.last_valid_velocity, self.held_velocity)

        position_noise = torch.zeros_like(valid_position)
        velocity_noise = torch.zeros_like(valid_velocity)
        if float(self.cfg.position_noise_std) > 0.0:
            position_noise = stateless_normal(self.env_ids, self.step_count, 3, self.seed, 101, self.dtype)
            position_noise = position_noise * float(self.cfg.position_noise_std)
        if float(self.cfg.velocity_noise_std) > 0.0:
            velocity_noise = stateless_normal(self.env_ids, self.step_count, 3, self.seed, 211, self.dtype)
            velocity_noise = velocity_noise * float(self.cfg.velocity_noise_std)

        position_obs = valid_position + position_noise
        velocity_obs = valid_velocity + velocity_noise
        self.last_position_obs[:] = position_obs
        self.last_velocity_obs[:] = velocity_obs
        self.last_position_update_mask[:] = position_update
        self.last_velocity_update_mask[:] = velocity_update
        self.last_position_dropout_mask[:] = position_dropout
        self.last_velocity_dropout_mask[:] = velocity_dropout
        self.last_position_read_age_steps[:] = torch.minimum(
            self.step_count, torch.full_like(self.step_count, int(self.cfg.position_delay_steps))
        )
        self.last_velocity_read_age_steps[:] = torch.minimum(
            self.step_count, torch.full_like(self.step_count, int(self.cfg.velocity_delay_steps))
        )
        self.observation_count += 1
        self.position_update_count += position_update.to(dtype=torch.long)
        self.velocity_update_count += velocity_update.to(dtype=torch.long)
        self.position_dropout_count += position_dropout.to(dtype=torch.long)
        self.velocity_dropout_count += velocity_dropout.to(dtype=torch.long)
        self.step_count += 1
        return position_obs, velocity_obs

    def runtime_state(self) -> dict[str, torch.Tensor]:
        """Return a cloned state snapshot for tests and audits."""

        return {
            "position_history": self.position_history.data.clone(),
            "velocity_history": self.velocity_history.data.clone(),
            "step_count": self.step_count.clone(),
            "held_position": self.held_position.clone(),
            "held_velocity": self.held_velocity.clone(),
            "last_valid_position": self.last_valid_position.clone(),
            "last_valid_velocity": self.last_valid_velocity.clone(),
            "last_position_obs": self.last_position_obs.clone(),
            "last_velocity_obs": self.last_velocity_obs.clone(),
            "observation_count": self.observation_count.clone(),
            "position_update_count": self.position_update_count.clone(),
            "velocity_update_count": self.velocity_update_count.clone(),
            "position_dropout_count": self.position_dropout_count.clone(),
            "velocity_dropout_count": self.velocity_dropout_count.clone(),
        }

    def diagnostics(self) -> dict[str, object]:
        """Return audit-only diagnostics that are never added to Actor observations."""

        observation_count = torch.clamp(self.observation_count.to(dtype=torch.float32), min=1.0)
        position_error = self.last_position_obs - self.last_position_truth
        velocity_error = self.last_velocity_obs - self.last_velocity_truth
        return {
            "cfg": copy.deepcopy(self.cfg).__dict__,
            "seed": int(self.seed),
            "history_length": int(self.cfg.history_length),
            "observation_count_min": int(self.observation_count.min().item()),
            "observation_count_max": int(self.observation_count.max().item()),
            "position_delay_age_min": int(self.last_position_read_age_steps.min().item()),
            "position_delay_age_max": int(self.last_position_read_age_steps.max().item()),
            "velocity_delay_age_min": int(self.last_velocity_read_age_steps.min().item()),
            "velocity_delay_age_max": int(self.last_velocity_read_age_steps.max().item()),
            "position_update_fraction_mean": float(torch.mean(self.position_update_count / observation_count).item()),
            "velocity_update_fraction_mean": float(torch.mean(self.velocity_update_count / observation_count).item()),
            "position_dropout_fraction_mean": float(torch.mean(self.position_dropout_count / observation_count).item()),
            "velocity_dropout_fraction_mean": float(torch.mean(self.velocity_dropout_count / observation_count).item()),
            "position_obs_truth_abs_max": float(torch.max(torch.abs(position_error)).item()),
            "velocity_obs_truth_abs_max": float(torch.max(torch.abs(velocity_error)).item()),
            "last_position_update_count": int(torch.count_nonzero(self.last_position_update_mask).item()),
            "last_velocity_update_count": int(torch.count_nonzero(self.last_velocity_update_mask).item()),
            "last_position_dropout_count": int(torch.count_nonzero(self.last_position_dropout_mask).item()),
            "last_velocity_dropout_count": int(torch.count_nonzero(self.last_velocity_dropout_mask).item()),
        }
