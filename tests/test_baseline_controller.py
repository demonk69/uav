"""Pure PyTorch tests for M4 deterministic baseline controller."""

import inspect

import torch

from uav_rendezvous_rl.controllers import compute_baseline_velocity_command, compute_limited_acceleration


def test_velocity_command_formula_is_translation_invariant() -> None:
    p_target_w = torch.tensor([[4.0, -2.0, 1.5]])
    v_target_w = torch.tensor([[0.4, 0.3, 0.0]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0]])
    p_ego_w = p_target_w + b_des_w + torch.tensor([[2.0, -1.0, 0.0]])
    translation_w = torch.tensor([[100.0, -50.0, 3.0]])

    v_cmd_w, *_ = compute_baseline_velocity_command(p_ego_w, p_target_w, v_target_w, b_des_w, 0.5, 10.0)
    shifted_v_cmd_w, *_ = compute_baseline_velocity_command(
        p_ego_w + translation_w, p_target_w + translation_w, v_target_w, b_des_w, 0.5, 10.0
    )

    torch.testing.assert_close(v_cmd_w, shifted_v_cmd_w)


def test_controller_signature_has_no_privileged_or_future_inputs() -> None:
    parameters = set(inspect.signature(compute_baseline_velocity_command).parameters)

    assert parameters == {"p_ego_w", "p_target_w", "v_target_w", "b_des_w", "prediction_horizon_s", "v_cmd_max"}
    assert "mode_id" not in parameters
    assert "a_target_w" not in parameters
    assert "future" not in " ".join(parameters)
    assert "schedule" not in " ".join(parameters)


def test_acceleration_command_tracks_velocity_error_with_limit() -> None:
    v_cmd_w = torch.tensor([[3.0, 4.0, 0.0], [0.1, 0.0, 0.0]])
    v_ego_w = torch.zeros_like(v_cmd_w)

    a_cmd_w, saturated = compute_limited_acceleration(v_cmd_w, v_ego_w, tau_v=0.25, a_max=2.0)

    torch.testing.assert_close(torch.linalg.norm(a_cmd_w[0]), torch.tensor(2.0))
    torch.testing.assert_close(a_cmd_w[1], torch.tensor([0.4, 0.0, 0.0]))
    assert torch.equal(saturated, torch.tensor([True, False]))
