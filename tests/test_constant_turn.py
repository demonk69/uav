"""Unit tests for M3 constant-turn target motion."""

import math

import torch

from uav_rendezvous_rl.motions.constant_turn import compute_constant_turn


def _independent_cv_reference(
    p_initial_w: torch.Tensor,
    v_initial_w: torch.Tensor,
    step_count: torch.Tensor,
    physics_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    elapsed_time = step_count.to(torch.float32).unsqueeze(-1) * physics_dt
    expected_position = p_initial_w + v_initial_w * elapsed_time
    expected_velocity = v_initial_w.clone()
    expected_acceleration = torch.zeros_like(v_initial_w)
    expected_position[:, 2] = p_initial_w[:, 2]
    expected_velocity[:, 2] = 0.0
    return expected_position, expected_velocity, expected_acceleration


def test_constant_turn_zero_omega_matches_constant_velocity() -> None:
    p_initial_w = torch.tensor([[1.0, 2.0, 1.5], [-4.0, 0.5, 2.0]])
    v_initial_w = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, -3.0]])
    omega = torch.zeros(2)
    step_count = torch.tensor([10, 25], dtype=torch.long)
    physics_dt = 0.02

    state = compute_constant_turn(p_initial_w, v_initial_w, omega, step_count, physics_dt, fixed_height=True)

    expected_position, expected_velocity, expected_acceleration = _independent_cv_reference(
        p_initial_w, v_initial_w, step_count, physics_dt
    )
    torch.testing.assert_close(state.p_target_w, expected_position)
    torch.testing.assert_close(state.v_target_w, expected_velocity)
    torch.testing.assert_close(state.a_target_w, expected_acceleration)


def test_constant_turn_near_zero_omega_uses_independent_straight_line_reference() -> None:
    p_initial_w = torch.tensor([[1.0, -2.0, 1.5], [-4.0, 0.5, 2.0]])
    v_initial_w = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, -3.0]])
    omega = torch.tensor([1.0e-7, -1.0e-7])
    step_count = torch.tensor([10, 25], dtype=torch.long)
    physics_dt = 0.02

    state = compute_constant_turn(
        p_initial_w,
        v_initial_w,
        omega,
        step_count,
        physics_dt,
        fixed_height=True,
        omega_epsilon=1.0e-5,
    )

    expected_position, expected_velocity, expected_acceleration = _independent_cv_reference(
        p_initial_w, v_initial_w, step_count, physics_dt
    )
    torch.testing.assert_close(state.p_target_w, expected_position)
    torch.testing.assert_close(state.v_target_w, expected_velocity)
    torch.testing.assert_close(state.a_target_w, expected_acceleration)


def test_constant_turn_quarter_turn_matches_closed_form() -> None:
    p_initial_w = torch.tensor([[0.0, 0.0, 1.5]])
    v_initial_w = torch.tensor([[1.0, 0.0, 0.0]])
    omega = torch.tensor([math.pi / 2.0])
    step_count = torch.tensor([1], dtype=torch.long)

    state = compute_constant_turn(p_initial_w, v_initial_w, omega, step_count, 1.0, fixed_height=True)

    expected_position = torch.tensor([[2.0 / math.pi, 2.0 / math.pi, 1.5]])
    expected_velocity = torch.tensor([[0.0, 1.0, 0.0]])
    expected_acceleration = torch.tensor([[-math.pi / 2.0, 0.0, 0.0]])
    torch.testing.assert_close(state.p_target_w, expected_position, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(state.v_target_w, expected_velocity, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(state.a_target_w, expected_acceleration, atol=1.0e-6, rtol=1.0e-6)
