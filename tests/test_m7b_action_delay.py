"""M7B action delay FIFO tests with independent expected values."""

import torch

from uav_rendezvous_rl.dynamics import ActionDelayBuffer


def test_fifo_impulse_delay_zero() -> None:
    buf = ActionDelayBuffer(num_envs=1, max_delay_steps=3, device="cpu")
    buf.reset(None)
    a = torch.tensor([[1.0, 2.0, 3.0]])
    delay = torch.tensor([0], dtype=torch.long)
    executed = buf.push_and_read(a, delay)
    assert torch.equal(executed, a)


def test_fifo_impulse_delay_one_returns_previous() -> None:
    buf = ActionDelayBuffer(num_envs=1, max_delay_steps=3, device="cpu")
    buf.reset(None)
    delay = torch.tensor([1], dtype=torch.long)
    a0 = torch.tensor([[1.0, 0.0, 0.0]])
    a1 = torch.tensor([[2.0, 0.0, 0.0]])
    e0 = buf.push_and_read(a0, delay)
    e1 = buf.push_and_read(a1, delay)
    assert torch.allclose(e0, torch.zeros_like(a0))
    assert torch.equal(e1, a0)


def test_fifo_impulse_delay_three_sequence() -> None:
    buf = ActionDelayBuffer(num_envs=1, max_delay_steps=3, device="cpu")
    buf.reset(None)
    delay = torch.tensor([3], dtype=torch.long)
    expected = []
    for step in range(6):
        a = torch.tensor([[float(step), 0.0, 0.0]])
        e = buf.push_and_read(a, delay)
        expected.append(float(e[0, 0].item()))
    assert expected == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]


def test_fifo_multienv_different_delays() -> None:
    buf = ActionDelayBuffer(num_envs=3, max_delay_steps=3, device="cpu")
    buf.reset(None)
    delay = torch.tensor([0, 1, 2], dtype=torch.long)
    a0 = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    a1 = torch.tensor([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    e0 = buf.push_and_read(a0, delay)
    e1 = buf.push_and_read(a1, delay)
    assert e0[0, 0].item() == 1.0
    assert e0[1, 0].item() == 0.0
    assert e0[2, 0].item() == 0.0
    assert e1[0, 0].item() == 4.0
    assert e1[1, 0].item() == 2.0
    assert e1[2, 0].item() == 0.0


def test_fifo_partial_reset_clears_selected_envs() -> None:
    buf = ActionDelayBuffer(num_envs=3, max_delay_steps=3, device="cpu")
    buf.reset(None)
    a = torch.ones((3, 3))
    delay = torch.tensor([0, 0, 0], dtype=torch.long)
    for _ in range(5):
        buf.push_and_read(a, delay)
    before = buf.buffer.clone()
    reset_ids = torch.tensor([1], dtype=torch.long)
    buf.reset(reset_ids)
    unselected = torch.tensor([0, 2], dtype=torch.long)
    assert not torch.allclose(buf.buffer[1], torch.ones(buf.buffer_depth, 3))
    assert torch.allclose(buf.buffer[unselected], before[unselected])
    full_reset_before = buf.buffer.clone()
    buf.reset(None)
    assert torch.allclose(buf.buffer, torch.zeros(3, buf.buffer_depth, 3))
