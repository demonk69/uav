"""Unit tests for M3 constant-acceleration target motion."""

import torch

from uav_rendezvous_rl.motions.constant_acceleration import compute_constant_acceleration


def test_constant_acceleration_matches_closed_form_solution() -> None:
    p_initial_w = torch.tensor([[1.0, 2.0, 1.5], [-4.0, 0.5, 2.0]])
    v_initial_w = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, -3.0]])
    acceleration_w = torch.tensor([[0.2, -0.1, 1.0], [-0.05, 0.15, -1.0]])
    step_count = torch.tensor([2, 5], dtype=torch.long)
    physics_dt = 0.1

    state = compute_constant_acceleration(
        p_initial_w, v_initial_w, acceleration_w, step_count, physics_dt, fixed_height=True
    )

    elapsed_time = step_count.to(torch.float32).unsqueeze(-1) * physics_dt
    expected_position = p_initial_w + v_initial_w * elapsed_time + 0.5 * acceleration_w * elapsed_time.square()
    expected_velocity = v_initial_w + acceleration_w * elapsed_time
    expected_acceleration = acceleration_w.clone()
    expected_position[:, 2] = p_initial_w[:, 2]
    expected_velocity[:, 2] = 0.0
    expected_acceleration[:, 2] = 0.0
    torch.testing.assert_close(state.p_target_w, expected_position)
    torch.testing.assert_close(state.v_target_w, expected_velocity)
    torch.testing.assert_close(state.a_target_w, expected_acceleration)
