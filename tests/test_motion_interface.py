"""Unit tests for the common M3 target motion interface."""

import torch

from uav_rendezvous_rl.motions import MotionState, TargetMotionManager, TargetMotionManagerCfg
from uav_rendezvous_rl.motions.sampling import make_split_generator


def test_target_motion_manager_reset_shapes() -> None:
    cfg = TargetMotionManagerCfg(force_mode_cycle_on_reset=True, max_acceleration=0.6)
    manager = TargetMotionManager(num_envs=4, cfg=cfg, physics_dt=0.01, device="cpu")
    generator = make_split_generator(123, cfg, "train")
    env_ids = torch.arange(4, dtype=torch.long)
    p_initial_w = torch.tensor(
        [[5.0, 0.0, 1.5], [6.0, 1.0, 1.5], [7.0, -1.0, 1.5], [8.0, 0.5, 1.5]]
    )
    v_initial_w = torch.tensor(
        [[0.5, 0.0, 0.0], [0.6, 0.1, 0.0], [0.7, -0.1, 0.0], [0.8, 0.2, 0.0]]
    )

    reset_state = manager.reset(env_ids, p_initial_w, v_initial_w, generator, "train")

    assert isinstance(reset_state, MotionState)
    assert reset_state.p_target_w.shape == (4, 3)
    assert reset_state.v_target_w.shape == (4, 3)
    assert reset_state.a_target_w.shape == (4, 3)
    torch.testing.assert_close(manager.mode_id, torch.tensor([0, 1, 2, 3]))
