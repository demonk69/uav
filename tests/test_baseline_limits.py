"""Pure PyTorch tests for M4 vector norm limits."""

import torch

from uav_rendezvous_rl.controllers import clamp_vector_norm, compute_baseline_velocity_command


def test_clamp_vector_norm_limits_magnitude_and_preserves_direction() -> None:
    vector_w = torch.tensor([[3.0, 4.0, 0.0], [0.3, 0.4, 0.0]])

    clamped_w, saturated = clamp_vector_norm(vector_w, 1.0)

    torch.testing.assert_close(clamped_w[0], torch.tensor([0.6, 0.8, 0.0]))
    torch.testing.assert_close(clamped_w[1], vector_w[1])
    assert torch.equal(saturated, torch.tensor([True, False]))


def test_velocity_command_norm_limit_applies_to_whole_vector() -> None:
    p_target_w = torch.zeros((1, 3))
    v_target_w = torch.zeros((1, 3))
    b_des_w = torch.tensor([[5.0, 0.0, 0.0]])
    p_ego_w = torch.tensor([[-5.0, -5.0, 0.0]])

    v_cmd_w, saturated, _p_pred_w, _p_goal_w = compute_baseline_velocity_command(
        p_ego_w, p_target_w, v_target_w, b_des_w, 0.5, 3.0
    )

    torch.testing.assert_close(torch.linalg.norm(v_cmd_w, dim=1), torch.tensor([3.0]))
    assert bool(saturated[0].item())
