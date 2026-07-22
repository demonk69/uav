"""Pure PyTorch tests for M4 control causality."""

import torch

from uav_rendezvous_rl.controllers import (
    compute_baseline_velocity_command,
    compute_limited_acceleration,
    integrate_ego_kinematics,
)


def test_control_uses_target_state_from_start_of_physics_substep() -> None:
    p_ego_w = torch.tensor([[8.0, 0.0, 1.5]])
    v_ego_w = torch.zeros((1, 3))
    p_target_start_w = torch.tensor([[1.0, 0.0, 1.5]])
    v_target_w = torch.tensor([[1.0, 0.0, 0.0]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0]])
    p_target_future_w = p_target_start_w + v_target_w * 0.01

    v_cmd_start_w, *_ = compute_baseline_velocity_command(
        p_ego_w, p_target_start_w, v_target_w, b_des_w, 0.5, 3.0
    )
    v_cmd_future_w, *_ = compute_baseline_velocity_command(
        p_ego_w, p_target_future_w, v_target_w, b_des_w, 0.5, 3.0
    )
    a_cmd_w, _ = compute_limited_acceleration(v_cmd_start_w, v_ego_w, tau_v=0.25, a_max=2.0)
    p_next_w, _v_next_w, _ = integrate_ego_kinematics(p_ego_w, v_ego_w, a_cmd_w, 0.01, 5.0)

    assert not torch.allclose(v_cmd_start_w, v_cmd_future_w)
    expected_v_cmd_w = v_target_w + (p_target_start_w + b_des_w - p_ego_w) / 0.5
    torch.testing.assert_close(v_cmd_start_w, expected_v_cmd_w)
    assert bool(torch.isfinite(p_next_w).all().item())
