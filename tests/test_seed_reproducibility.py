"""Unit tests for M2 seeded reset sampling."""

import torch

from uav_rendezvous_rl.tasks.direct.m2_kinematics import M2RandomizationCfg, sample_m2_initial_conditions


def _cfg() -> M2RandomizationCfg:
    return M2RandomizationCfg(
        ego_initial_pos_w=(0.0, 0.0, 1.5),
        target_pos_x_range=(4.0, 8.0),
        target_pos_y_range=(-2.0, 2.0),
        target_height_range=(1.5, 1.5),
        target_vel_x_range=(0.2, 1.0),
        target_vel_y_range=(-0.5, 0.5),
        d_safe=0.75,
    )


def test_same_seed_reproduces_initial_conditions() -> None:
    generator_a = torch.Generator(device="cpu").manual_seed(123)
    generator_b = torch.Generator(device="cpu").manual_seed(123)

    sample_a = sample_m2_initial_conditions(16, _cfg(), generator_a, "cpu")
    sample_b = sample_m2_initial_conditions(16, _cfg(), generator_b, "cpu")

    for tensor_a, tensor_b in zip(sample_a, sample_b, strict=True):
        torch.testing.assert_close(tensor_a, tensor_b)


def test_envs_receive_independent_target_randomization() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    _p_ego_w, _v_ego_w, p_target_initial_w, v_target_w = sample_m2_initial_conditions(16, _cfg(), generator, "cpu")

    assert torch.unique(p_target_initial_w[:, :2], dim=0).shape[0] > 1
    assert torch.unique(v_target_w[:, :2], dim=0).shape[0] > 1
    assert torch.all(torch.linalg.norm(p_target_initial_w - _p_ego_w, dim=1) > _cfg().d_safe)
