"""Unit tests for M2 fixed-height constant-velocity target motion."""

import torch

from uav_rendezvous_rl.tasks.direct.m2_kinematics import compute_constant_velocity_position_w


def test_constant_velocity_motion_matches_analytic_solution() -> None:
    p_initial_w = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 0.5, 1.5]])
    v_w = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, -3.0]])
    elapsed_time = torch.tensor([2.0, 4.0])

    p_w = compute_constant_velocity_position_w(p_initial_w, v_w, elapsed_time)
    expected = p_initial_w + v_w * elapsed_time.unsqueeze(-1)
    expected[:, 2] = p_initial_w[:, 2]

    torch.testing.assert_close(p_w, expected)


def test_constant_velocity_keeps_height_fixed() -> None:
    p_initial_w = torch.tensor([[0.0, 0.0, 1.25], [1.0, -1.0, 2.0]])
    v_w = torch.tensor([[0.0, 0.0, 10.0], [0.5, 0.25, -10.0]])
    elapsed_time = torch.tensor([0.5, 1.5])

    p_w = compute_constant_velocity_position_w(p_initial_w, v_w, elapsed_time)

    torch.testing.assert_close(p_w[:, 2], p_initial_w[:, 2])
