"""Unit tests for fixed M2 truth-state definitions."""

import torch

from uav_rendezvous_rl.tasks.direct.m2_kinematics import compute_offset_error_w, compute_relative_state_w


def test_relative_state_signs_match_project_definitions() -> None:
    p_ego_w = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.5]])
    v_ego_w = torch.tensor([[0.5, -1.0, 0.0], [0.0, 0.25, -0.5]])
    p_target_w = torch.tensor([[4.0, 1.0, 5.0], [-3.0, 2.0, 1.0]])
    v_target_w = torch.tensor([[-0.5, 2.0, 0.25], [1.5, -0.25, 0.0]])

    p_rel_w, v_rel_w = compute_relative_state_w(p_ego_w, v_ego_w, p_target_w, v_target_w)

    torch.testing.assert_close(p_rel_w, p_target_w - p_ego_w)
    torch.testing.assert_close(v_rel_w, v_target_w - v_ego_w)


def test_offset_error_sign_matches_project_definition() -> None:
    p_ego_w = torch.tensor([[6.0, 1.0, 2.0], [2.0, -1.0, 1.5]])
    p_target_w = torch.tensor([[1.0, 1.0, 2.0], [-3.0, -1.0, 1.5]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    e_offset_w = compute_offset_error_w(p_ego_w, p_target_w, b_des_w)

    torch.testing.assert_close(e_offset_w, p_ego_w - p_target_w - b_des_w)
    torch.testing.assert_close(e_offset_w[0], torch.zeros(3))
