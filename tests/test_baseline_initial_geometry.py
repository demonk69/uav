"""Pure PyTorch tests for M4 baseline initial geometry."""

import torch

from uav_rendezvous_rl.controllers import (
    BaselineInitialGeometryCfg,
    sample_baseline_initial_ego_state,
    validate_baseline_initial_geometry,
)


def test_initial_geometry_samples_ego_outside_desired_offset_point() -> None:
    cfg = BaselineInitialGeometryCfg(delta_x_range=(2.0, 6.0), delta_y_range=(-2.0, 2.0), delta_z=0.0)
    generator = torch.Generator(device="cpu").manual_seed(11)
    p_target_w = torch.tensor([[4.0, 0.0, 1.5], [6.0, 1.0, 1.5]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    p_ego_w, v_ego_w, delta_w = sample_baseline_initial_ego_state(p_target_w, b_des_w, cfg, generator, "cpu")
    geometry = validate_baseline_initial_geometry(p_target_w, p_ego_w, b_des_w, d_safe=0.75, min_initial_offset_error=1.0)

    assert bool(torch.all(delta_w[:, 0] >= 2.0).item())
    assert bool(torch.all(delta_w[:, 0] <= 6.0).item())
    assert bool(torch.all(delta_w[:, 1] >= -2.0).item())
    assert bool(torch.all(delta_w[:, 1] <= 2.0).item())
    torch.testing.assert_close(delta_w[:, 2], torch.zeros(2))
    torch.testing.assert_close(v_ego_w, torch.zeros_like(v_ego_w))
    torch.testing.assert_close(p_ego_w, p_target_w + b_des_w + delta_w)
    assert bool(torch.all(geometry["valid"]).item())
    assert bool(torch.all(geometry["segment_distance"] > 0.75).item())


def test_initial_geometry_rejects_path_through_target_safety_sphere() -> None:
    p_target_w = torch.tensor([[0.0, 0.0, 1.5]])
    b_des_w = torch.tensor([[0.5, 0.0, 0.0]])
    p_ego_w = torch.tensor([[-0.5, 0.0, 1.5]])

    geometry = validate_baseline_initial_geometry(p_target_w, p_ego_w, b_des_w, d_safe=0.75, min_initial_offset_error=0.1)

    assert not bool(geometry["valid"][0].item())
    assert geometry["segment_distance"][0].item() < 0.75
