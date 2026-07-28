"""Per-environment action delay FIFO for M7B control execution delay."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class ActionDelayBuffer:
    """Push-then-read FIFO storing squashed actions (shape (num_envs, max_delay+1, 3))."""

    def __init__(self, num_envs: int, max_delay_steps: int, device: torch.device | str, dtype: torch.dtype = torch.float32):
        if int(max_delay_steps) < 0:
            raise ValueError("max_delay_steps must be non-negative.")
        self.num_envs = int(num_envs)
        self.max_delay_steps = int(max_delay_steps)
        self.buffer_depth = self.max_delay_steps + 1
        self.device = torch.device(device)
        self.dtype = dtype
        self.buffer = torch.zeros((self.num_envs, self.buffer_depth, 3), dtype=dtype, device=self.device)
        self.write_index = 0

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        ids = self._resolve_env_ids(env_ids)
        self.buffer[ids] = 0.0

    def push_and_read(self, squashed_action: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
        self.buffer[:, self.write_index, :] = squashed_action.to(device=self.device, dtype=self.dtype)
        read_indices = (self.write_index - delay_steps.to(device=self.device, dtype=torch.long)) % self.buffer_depth
        env_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        executed = self.buffer[env_indices, read_indices, :]
        self.write_index = (self.write_index + 1) % self.buffer_depth
        return executed
