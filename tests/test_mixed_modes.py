"""Unit tests for mixed per-env M3 target motion modes."""

from dataclasses import replace

import torch

from uav_rendezvous_rl.motions import TargetMotionManager, TargetMotionManagerCfg
from uav_rendezvous_rl.motions.sampling import make_split_generator


def test_manager_selects_independent_mixed_modes_per_env() -> None:
    base_cfg = TargetMotionManagerCfg(force_mode_cycle_on_reset=True, max_speed=20.0, max_acceleration=0.6)
    train_cfg = replace(
        base_cfg.train,
        acceleration_x_range=(0.1, 0.1),
        acceleration_y_range=(0.0, 0.0),
        turn_omega_range=(0.5, 0.5),
        piecewise_acceleration_x_range=(0.2, 0.2),
        piecewise_acceleration_y_range=(0.0, 0.0),
        piecewise_segment_duration_steps_range=(10, 10),
    )
    cfg = replace(base_cfg, train=train_cfg)
    manager = TargetMotionManager(num_envs=4, cfg=cfg, physics_dt=0.1, device="cpu")
    generator = make_split_generator(9, cfg, "train")
    env_ids = torch.arange(4, dtype=torch.long)
    p_initial_w = torch.tensor([[5.0, 0.0, 1.5]]).repeat(4, 1)
    v_initial_w = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)

    manager.reset(env_ids, p_initial_w, v_initial_w, generator, "train")
    state = manager.step()

    torch.testing.assert_close(manager.mode_id, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(state.a_target_w[0], torch.zeros(3))
    torch.testing.assert_close(state.a_target_w[1], torch.tensor([0.1, 0.0, 0.0]))
    assert torch.linalg.norm(state.a_target_w[2, :2]).item() > 0.0
    torch.testing.assert_close(state.a_target_w[3], torch.tensor([0.2, 0.0, 0.0]))
    assert bool(torch.isfinite(state.p_target_w).all().item())
