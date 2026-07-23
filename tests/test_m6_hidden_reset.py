"""Unit tests for RSL-RL GRU hidden-state reset semantics used by M6."""

import torch
from rsl_rl.networks import Memory


def test_gru_memory_done_mask_clears_done_envs_only() -> None:
    memory = Memory(input_size=25, hidden_dim=128, num_layers=1, type="gru")
    memory(torch.zeros((4, 25), dtype=torch.float32))
    memory.hidden_state[:] = 1.0

    dones = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    memory.reset(dones)

    assert memory.hidden_state.shape == (1, 4, 128)
    assert torch.all(memory.hidden_state[:, dones == 1, :] == 0.0)
    assert torch.all(memory.hidden_state[:, dones == 0, :] == 1.0)


def test_gru_memory_full_reset_clears_hidden_state() -> None:
    memory = Memory(input_size=57, hidden_dim=128, num_layers=1, type="gru")
    memory(torch.zeros((2, 57), dtype=torch.float32))

    assert memory.hidden_state is not None
    memory.reset()

    assert memory.hidden_state is None
