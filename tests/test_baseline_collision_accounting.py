"""Pure PyTorch tests for M4 collision-risk episode accounting."""

import torch

from uav_rendezvous_rl.tasks.direct.m4_accounting import update_collision_risk_accounting


def test_first_collision_increments_count_once() -> None:
    collision_risk_buf = torch.zeros(1, dtype=torch.bool)
    collision_risk_count = torch.zeros(1, dtype=torch.long)

    new_collision = update_collision_risk_accounting(torch.tensor([True]), collision_risk_buf, collision_risk_count)

    assert torch.equal(new_collision, torch.tensor([True]))
    assert torch.equal(collision_risk_buf, torch.tensor([True]))
    assert torch.equal(collision_risk_count, torch.tensor([1]))


def test_persistent_collision_does_not_increment_again() -> None:
    collision_risk_buf = torch.zeros(1, dtype=torch.bool)
    collision_risk_count = torch.zeros(1, dtype=torch.long)

    for _ in range(5):
        update_collision_risk_accounting(torch.tensor([True]), collision_risk_buf, collision_risk_count)

    assert torch.equal(collision_risk_buf, torch.tensor([True]))
    assert torch.equal(collision_risk_count, torch.tensor([1]))


def test_reset_clears_episode_state_and_allows_one_new_count() -> None:
    collision_risk_buf = torch.zeros(1, dtype=torch.bool)
    collision_risk_count = torch.zeros(1, dtype=torch.long)
    update_collision_risk_accounting(torch.tensor([True]), collision_risk_buf, collision_risk_count)

    collision_risk_buf[:] = False
    collision_risk_count[:] = 0
    update_collision_risk_accounting(torch.tensor([True]), collision_risk_buf, collision_risk_count)

    assert torch.equal(collision_risk_buf, torch.tensor([True]))
    assert torch.equal(collision_risk_count, torch.tensor([1]))


def test_no_collision_keeps_count_zero() -> None:
    collision_risk_buf = torch.zeros(3, dtype=torch.bool)
    collision_risk_count = torch.zeros(3, dtype=torch.long)

    for _ in range(5):
        new_collision = update_collision_risk_accounting(
            torch.tensor([False, False, False]), collision_risk_buf, collision_risk_count
        )

    assert torch.equal(new_collision, torch.tensor([False, False, False]))
    assert torch.equal(collision_risk_buf, torch.tensor([False, False, False]))
    assert torch.equal(collision_risk_count, torch.tensor([0, 0, 0]))
