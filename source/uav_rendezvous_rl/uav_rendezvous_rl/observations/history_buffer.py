"""Vectorized fixed-length observation history buffer."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class ObservationHistoryBuffer:
    """Store current and past samples with explicit delay semantics."""

    def __init__(
        self,
        num_envs: int,
        dim: int,
        history_length: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ):
        if int(history_length) < 1:
            raise ValueError("history_length must be at least 1.")
        self.num_envs = int(num_envs)
        self.dim = int(dim)
        self.history_length = int(history_length)
        self.device = torch.device(device)
        self.dtype = dtype
        self.data = torch.zeros((self.num_envs, self.history_length, self.dim), dtype=dtype, device=self.device)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None, sample: torch.Tensor) -> None:
        """Fill selected environments' entire history with the current initial sample."""

        ids = self._resolve_env_ids(env_ids)
        values = sample.to(device=self.device, dtype=self.dtype)
        if values.shape != (ids.numel(), self.dim):
            raise ValueError(f"Expected reset sample shape {(ids.numel(), self.dim)}, got {tuple(values.shape)}.")
        self.data[ids] = values.unsqueeze(1).expand(-1, self.history_length, -1)

    def push(self, sample: torch.Tensor) -> None:
        """Append a current sample for all environments."""

        values = sample.to(device=self.device, dtype=self.dtype)
        if values.shape != (self.num_envs, self.dim):
            raise ValueError(f"Expected sample shape {(self.num_envs, self.dim)}, got {tuple(values.shape)}.")
        if self.history_length > 1:
            self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = values

    def read(self, delay_steps: int) -> torch.Tensor:
        """Return current sample for delay 0, previous sample for delay 1, and so on."""

        delay = int(delay_steps)
        if delay < 0 or delay >= self.history_length:
            raise ValueError(f"delay_steps={delay} outside history length {self.history_length}.")
        return self.data[:, self.history_length - 1 - delay]
