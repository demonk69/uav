"""Unit tests for M3 partial target motion reset semantics."""

import torch

from uav_rendezvous_rl.motions import TargetMotionManager, TargetMotionManagerCfg
from uav_rendezvous_rl.motions.sampling import make_split_generator


def _assert_tensor_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.dtype.is_floating_point:
        torch.testing.assert_close(left, right)
    else:
        assert torch.equal(left, right)


def test_partial_reset_replaces_only_selected_motion_states() -> None:
    cfg = TargetMotionManagerCfg(force_mode_cycle_on_reset=True, max_acceleration=0.6)
    manager = TargetMotionManager(num_envs=5, cfg=cfg, physics_dt=0.01, device="cpu")
    generator = make_split_generator(17, cfg, "train")
    all_env_ids = torch.arange(5, dtype=torch.long)
    p_initial_w = torch.tensor(
        [[5.0, 0.0, 1.5], [6.0, 1.0, 1.5], [7.0, -1.0, 1.5], [8.0, 0.5, 1.5], [9.0, -0.5, 1.5]]
    )
    v_initial_w = torch.tensor(
        [[0.5, 0.0, 0.0], [0.6, 0.1, 0.0], [0.7, -0.1, 0.0], [0.8, 0.2, 0.0], [0.9, -0.2, 0.0]]
    )
    manager.reset(all_env_ids, p_initial_w, v_initial_w, generator, "train")
    manager.step()
    manager.step()

    field_names = (
        "p0_w",
        "v0_w",
        "mode_id",
        "motion_step_count",
        "constant_acceleration_w",
        "turn_omega",
        "segment_index",
        "segment_step_count",
        "segment_duration_steps",
        "segment_start_position_w",
        "segment_start_velocity_w",
        "current_acceleration_w",
        "segment_switch_count",
    )
    before = {name: getattr(manager, name).detach().clone() for name in field_names}
    reset_env_ids = torch.tensor([1, 3], dtype=torch.long)
    untouched_env_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    new_positions_w = torch.tensor([[10.0, 2.0, 1.5], [11.0, -2.0, 1.5]])
    new_velocities_w = torch.tensor([[0.3, 0.4, 0.0], [0.2, -0.3, 0.0]])

    manager.reset(reset_env_ids, new_positions_w, new_velocities_w, generator, "train")

    for name in field_names:
        _assert_tensor_equal(getattr(manager, name)[untouched_env_ids], before[name][untouched_env_ids])
    torch.testing.assert_close(manager.p0_w[reset_env_ids], new_positions_w)
    torch.testing.assert_close(manager.v0_w[reset_env_ids], new_velocities_w)
    torch.testing.assert_close(manager.motion_step_count[reset_env_ids], torch.zeros(2, dtype=torch.long))
