"""M7B partial reset invariants for FIFO and stateless RNG."""

import torch

from uav_rendezvous_rl.dynamics import ActionDelayBuffer, stateless_m7b_uniform


def test_partial_reset_does_not_change_unselected_fifo_rows() -> None:
    buffer = ActionDelayBuffer(num_envs=4, max_delay_steps=3, device="cpu")
    for step in range(4):
        actions = torch.full((4, 3), float(step + 1))
        buffer.push_and_read(actions, torch.zeros(4, dtype=torch.long))
    before = buffer.buffer.clone()

    buffer.reset(torch.tensor([1, 3], dtype=torch.long))

    assert torch.allclose(buffer.buffer[0], before[0])
    assert torch.allclose(buffer.buffer[2], before[2])
    assert torch.allclose(buffer.buffer[1], torch.zeros_like(buffer.buffer[1]))
    assert torch.allclose(buffer.buffer[3], torch.zeros_like(buffer.buffer[3]))


def test_partial_reset_rng_future_for_unselected_env_is_unchanged() -> None:
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    episodes_before = torch.tensor([5, 5, 5], dtype=torch.long)
    episodes_after_env1_reset = torch.tensor([5, 6, 5], dtype=torch.long)

    before = stateless_m7b_uniform(env_ids, episodes_before, base_seed=123, stream=1001)
    after = stateless_m7b_uniform(env_ids, episodes_after_env1_reset, base_seed=123, stream=1001)

    assert torch.allclose(before[0], after[0])
    assert not torch.allclose(before[1], after[1])
    assert torch.allclose(before[2], after[2])
