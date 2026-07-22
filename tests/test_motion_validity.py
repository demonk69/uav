"""Unit tests for M3 target motion validity checks."""

import torch

from uav_rendezvous_rl.motions import MotionState, TargetMotionManager, TargetMotionManagerCfg
from uav_rendezvous_rl.motions.sampling import make_split_generator


def test_reset_motion_state_is_valid() -> None:
    cfg = TargetMotionManagerCfg(force_mode_cycle_on_reset=True, max_acceleration=0.6)
    manager = TargetMotionManager(num_envs=4, cfg=cfg, physics_dt=0.01, device="cpu")
    generator = make_split_generator(5, cfg, "train")
    p_initial_w = torch.tensor([[5.0, 0.0, 1.5]]).repeat(4, 1)
    v_initial_w = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)

    manager.reset(torch.arange(4), p_initial_w, v_initial_w, generator, "train")

    assert not bool(manager.invalid_mask.any().item())


def test_invalid_state_flags_non_finite_and_limit_violations() -> None:
    manager = TargetMotionManager(num_envs=3, cfg=TargetMotionManagerCfg(), physics_dt=0.01, device="cpu")
    manager.segment_duration_steps[:] = 1
    valid_position_w = torch.tensor([[5.0, 0.0, 1.5], [6.0, 0.0, 1.5], [7.0, 0.0, 1.5]])
    invalid_state = MotionState(
        p_target_w=valid_position_w,
        v_target_w=torch.tensor([[9.0, 0.0, 0.0], [0.5, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
        a_target_w=torch.zeros((3, 3)),
    )

    invalid_mask = manager._validate_state(invalid_state)

    assert torch.equal(invalid_mask, torch.tensor([True, False, True]))
