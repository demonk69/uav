"""Pure PyTorch tests for M5 randomized desired-offset geometry."""

import torch

from uav_rendezvous_rl.mdp import RendezvousInitialGeometryCfg, sample_m5_initial_geometry, validate_m5_initial_geometry


def test_m5_initial_geometry_samples_horizontal_random_offset() -> None:
    cfg = RendezvousInitialGeometryCfg(
        delta_radial_range=(2.0, 6.0), delta_tangent_range=(-2.0, 2.0), max_resample_attempts=4
    )
    generator = torch.Generator(device="cpu").manual_seed(5)
    p_target_w = torch.tensor([[4.0, 0.0, 1.5], [6.0, 1.0, 1.5], [5.0, -1.0, 1.5]])

    geometry = sample_m5_initial_geometry(p_target_w, d_offset=5.0, d_safe=0.75, cfg=cfg, generator=generator, device="cpu")
    validity = validate_m5_initial_geometry(
        p_target_w, geometry.p_ego_w, geometry.b_des_w, d_safe=0.75, min_initial_offset_error=1.0
    )

    torch.testing.assert_close(torch.linalg.norm(geometry.b_des_w, dim=1), torch.full((3,), 5.0), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(geometry.b_des_w[:, 2], torch.zeros(3))
    assert bool(torch.all(geometry.delta_radial >= 2.0).item())
    assert bool(torch.all(geometry.delta_radial <= 6.0).item())
    assert bool(torch.all(geometry.delta_tangent >= -2.0).item())
    assert bool(torch.all(geometry.delta_tangent <= 2.0).item())
    assert bool(torch.all(validity["valid"]).item())
    assert bool(torch.all(validity["center_distance"] > 0.75).item())


def test_m5_initial_geometry_is_seed_reproducible() -> None:
    cfg = RendezvousInitialGeometryCfg(max_resample_attempts=4)
    p_target_w = torch.tensor([[4.0, 0.0, 1.5], [6.0, 1.0, 1.5]])
    first_generator = torch.Generator(device="cpu").manual_seed(123)
    second_generator = torch.Generator(device="cpu").manual_seed(123)

    first = sample_m5_initial_geometry(p_target_w, 5.0, 0.75, cfg, first_generator, "cpu")
    second = sample_m5_initial_geometry(p_target_w, 5.0, 0.75, cfg, second_generator, "cpu")

    torch.testing.assert_close(first.p_ego_w, second.p_ego_w)
    torch.testing.assert_close(first.b_des_w, second.b_des_w)
    torch.testing.assert_close(first.theta, second.theta)
