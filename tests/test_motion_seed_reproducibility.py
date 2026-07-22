"""Unit tests for M3 seeded target motion reproducibility."""

import torch

from uav_rendezvous_rl.motions import TargetMotionManager, TargetMotionManagerCfg
from uav_rendezvous_rl.motions.configs import get_split_cfg
from uav_rendezvous_rl.motions.sampling import make_split_generator, sample_initial_target_state


def _sample_motion(seed: int) -> dict[str, torch.Tensor]:
    cfg = TargetMotionManagerCfg()
    generator = make_split_generator(seed, cfg, "train")
    num_envs = 16
    p_ego_w = torch.tensor([[0.0, 0.0, 1.5]]).repeat(num_envs, 1)
    p_initial_w, v_initial_w = sample_initial_target_state(
        num_envs, p_ego_w, cfg, get_split_cfg(cfg, "train"), generator, "cpu"
    )
    manager = TargetMotionManager(num_envs=num_envs, cfg=cfg, physics_dt=0.01, device="cpu")
    manager.reset(torch.arange(num_envs), p_initial_w, v_initial_w, generator, "train")
    for _ in range(20):
        state = manager.step()
    return {
        "p_initial_w": p_initial_w,
        "v_initial_w": v_initial_w,
        "mode_id": manager.mode_id.detach().clone(),
        "constant_acceleration_w": manager.constant_acceleration_w.detach().clone(),
        "turn_omega": manager.turn_omega.detach().clone(),
        "segment_duration_steps": manager.segment_duration_steps.detach().clone(),
        "current_acceleration_w": manager.current_acceleration_w.detach().clone(),
        "p_target_w": state.p_target_w.detach().clone(),
        "v_target_w": state.v_target_w.detach().clone(),
        "a_target_w": state.a_target_w.detach().clone(),
    }


def test_same_seed_reproduces_motion_parameters_and_state() -> None:
    sample_a = _sample_motion(123)
    sample_b = _sample_motion(123)

    for key in sample_a:
        if sample_a[key].dtype.is_floating_point:
            torch.testing.assert_close(sample_a[key], sample_b[key])
        else:
            assert torch.equal(sample_a[key], sample_b[key])


def test_different_seed_changes_sampled_motion() -> None:
    sample_a = _sample_motion(123)
    sample_b = _sample_motion(124)

    assert any(not torch.equal(sample_a[key], sample_b[key]) for key in sample_a)
