"""Partial done-mask regression tests for independent actor and critic GRU memories."""

import torch
from rsl_rl.networks import Memory


def test_actor_and_critic_memories_reset_independently_with_same_done_mask() -> None:
    memory_a = Memory(input_size=25, hidden_dim=128, num_layers=1, type="gru")
    memory_c = Memory(input_size=57, hidden_dim=128, num_layers=1, type="gru")
    memory_a(torch.ones((3, 25), dtype=torch.float32))
    memory_c(torch.ones((3, 57), dtype=torch.float32))
    memory_a.hidden_state[:] = 2.0
    memory_c.hidden_state[:] = 3.0

    dones = torch.tensor([1, 0, 1], dtype=torch.long)
    memory_a.reset(dones)
    memory_c.reset(dones)

    assert memory_a is not memory_c
    assert torch.all(memory_a.hidden_state[:, dones == 1, :] == 0.0)
    assert torch.all(memory_c.hidden_state[:, dones == 1, :] == 0.0)
    assert torch.all(memory_a.hidden_state[:, dones == 0, :] == 2.0)
    assert torch.all(memory_c.hidden_state[:, dones == 0, :] == 3.0)
