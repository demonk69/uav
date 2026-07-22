"""Pure PyTorch tests for M4 target prediction and goal formulas."""

import torch

from uav_rendezvous_rl.controllers import compute_baseline_velocity_command, compute_target_prediction


def test_target_prediction_matches_current_state_constant_velocity_formula() -> None:
    p_target_w = torch.tensor([[1.0, 2.0, 1.5], [-3.0, 0.5, 1.5]])
    v_target_w = torch.tensor([[0.5, -0.25, 0.0], [1.0, 0.75, 0.0]])

    p_pred_w = compute_target_prediction(p_target_w, v_target_w, 0.5)

    torch.testing.assert_close(p_pred_w, p_target_w + v_target_w * 0.5)


def test_goal_and_velocity_command_use_single_target_velocity_compensation() -> None:
    p_target_w = torch.tensor([[2.0, -1.0, 1.5]])
    v_target_w = torch.tensor([[0.8, 0.2, 0.0]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0]])
    p_ego_w = p_target_w + b_des_w


    v_cmd_w, saturated, p_pred_w, p_goal_w = compute_baseline_velocity_command(
        p_ego_w, p_target_w, v_target_w, b_des_w, 0.5, 3.0
    )

    torch.testing.assert_close(p_pred_w, p_target_w + v_target_w * 0.5)
    torch.testing.assert_close(p_goal_w, p_pred_w + b_des_w)
    torch.testing.assert_close(v_cmd_w, v_target_w)
    assert not bool(saturated.any().item())
