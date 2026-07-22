"""Unit tests for fixed-height constant-velocity target motion."""

import torch

from uav_rendezvous_rl.motions.constant_velocity import compute_constant_velocity
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


def test_m3_constant_velocity_returns_velocity_and_zero_acceleration() -> None:
    p_initial_w = torch.tensor([[1.0, 2.0, 1.5], [-4.0, 0.5, 2.0]])
    v_w = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, -3.0]])
    step_count = torch.tensor([10, 25], dtype=torch.long)
    physics_dt = 0.02

    state = compute_constant_velocity(p_initial_w, v_w, step_count, physics_dt, fixed_height=True)

    elapsed_time = step_count.to(torch.float32).unsqueeze(-1) * physics_dt
    expected_position = p_initial_w + v_w * elapsed_time
    expected_position[:, 2] = p_initial_w[:, 2]
    expected_velocity = v_w.clone()
    expected_velocity[:, 2] = 0.0
    expected_acceleration = torch.zeros_like(v_w)
    torch.testing.assert_close(state.p_target_w, expected_position)
    torch.testing.assert_close(state.v_target_w, expected_velocity)
    torch.testing.assert_close(state.a_target_w, expected_acceleration)
