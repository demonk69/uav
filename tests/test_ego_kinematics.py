"""Pure PyTorch tests for M4 ego kinematics."""

import torch

from uav_rendezvous_rl.controllers import integrate_ego_kinematics


def test_ego_integration_matches_constant_acceleration_equations() -> None:
    p_ego_w = torch.tensor([[1.0, 2.0, 1.5]])
    v_ego_w = torch.tensor([[0.5, -0.25, 0.0]])
    a_cmd_w = torch.tensor([[1.0, 0.5, 0.0]])

    p_next_w, v_next_w, saturated = integrate_ego_kinematics(p_ego_w, v_ego_w, a_cmd_w, 0.1, 5.0)

    torch.testing.assert_close(p_next_w, p_ego_w + v_ego_w * 0.1 + 0.5 * a_cmd_w * 0.01)
    torch.testing.assert_close(v_next_w, v_ego_w + a_cmd_w * 0.1)
    assert not bool(saturated.any().item())


def test_ego_velocity_hard_limit_uses_vector_norm() -> None:
    p_ego_w = torch.zeros((1, 3))
    v_ego_w = torch.tensor([[4.9, 0.0, 0.0]])
    a_cmd_w = torch.tensor([[10.0, 0.0, 0.0]])

    _p_next_w, v_next_w, saturated = integrate_ego_kinematics(p_ego_w, v_ego_w, a_cmd_w, 0.1, 5.0)

    torch.testing.assert_close(torch.linalg.norm(v_next_w, dim=1), torch.tensor([5.0]))
    assert bool(saturated[0].item())
