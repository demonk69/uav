"""Pure PyTorch tests for the M5 raw-action velocity mapping."""

import torch

from uav_rendezvous_rl.mdp import map_raw_action_to_velocity_command, raw_action_from_velocity_command


def test_raw_action_maps_through_tanh_to_velocity_command() -> None:
    raw_action = torch.tensor([[0.0, 1.0, -1.0], [10.0, -10.0, 0.5]])

    squashed, v_cmd_w = map_raw_action_to_velocity_command(raw_action, v_max=3.0)

    torch.testing.assert_close(squashed, torch.tanh(raw_action))
    torch.testing.assert_close(v_cmd_w, 3.0 * torch.tanh(raw_action))
    assert bool(torch.all(torch.abs(squashed) <= 1.0).item())
    assert bool(torch.all(torch.abs(v_cmd_w) <= 3.0).item())


def test_velocity_command_inverts_to_raw_action_inside_bounds() -> None:
    v_cmd_w = torch.tensor([[0.0, 1.5, -1.5], [2.0, -2.5, 0.5]])

    raw_action = raw_action_from_velocity_command(v_cmd_w, v_max=3.0)
    _squashed, reconstructed_v_cmd_w = map_raw_action_to_velocity_command(raw_action, v_max=3.0)

    torch.testing.assert_close(reconstructed_v_cmd_w, v_cmd_w, atol=1.0e-6, rtol=1.0e-6)
