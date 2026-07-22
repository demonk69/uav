"""Unit tests for partial M2 truth-state replacement semantics."""

import torch

from uav_rendezvous_rl.tasks.direct.m2_kinematics import M2RandomizationCfg, sample_m2_initial_conditions


def test_partial_reset_replaces_only_selected_env_truth_states() -> None:
    cfg = M2RandomizationCfg(
        ego_initial_pos_w=(0.0, 0.0, 1.5),
        target_pos_x_range=(4.0, 8.0),
        target_pos_y_range=(-2.0, 2.0),
        target_height_range=(1.5, 1.5),
        target_vel_x_range=(0.2, 1.0),
        target_vel_y_range=(-0.5, 0.5),
        d_safe=0.75,
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    env_ids = torch.tensor([1, 3], dtype=torch.long)

    p_ego_w = torch.full((5, 3), -10.0)
    v_ego_w = torch.full((5, 3), -20.0)
    p_target_initial_w = torch.full((5, 3), -30.0)
    v_target_w = torch.full((5, 3), -40.0)
    originals = tuple(tensor.clone() for tensor in (p_ego_w, v_ego_w, p_target_initial_w, v_target_w))

    reset_values = sample_m2_initial_conditions(len(env_ids), cfg, generator, "cpu")
    for tensor, values in zip((p_ego_w, v_ego_w, p_target_initial_w, v_target_w), reset_values, strict=True):
        tensor[env_ids] = values

    untouched_env_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    for tensor, original in zip((p_ego_w, v_ego_w, p_target_initial_w, v_target_w), originals, strict=True):
        torch.testing.assert_close(tensor[untouched_env_ids], original[untouched_env_ids])

    for tensor, values in zip((p_ego_w, v_ego_w, p_target_initial_w, v_target_w), reset_values, strict=True):
        torch.testing.assert_close(tensor[env_ids], values)
