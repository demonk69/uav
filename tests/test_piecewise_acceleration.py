"""Unit tests for M3 piecewise-acceleration target motion."""

from dataclasses import replace

import torch

from uav_rendezvous_rl.motions import TargetMotionManager, TargetMotionManagerCfg
from uav_rendezvous_rl.motions.piecewise_acceleration import compute_piecewise_acceleration
from uav_rendezvous_rl.motions.sampling import make_split_generator


def _independent_exact_accel_step(
    position_w: torch.Tensor,
    velocity_w: torch.Tensor,
    acceleration_w: torch.Tensor,
    physics_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_position_w = position_w + velocity_w * physics_dt + 0.5 * acceleration_w * physics_dt * physics_dt
    next_velocity_w = velocity_w + acceleration_w * physics_dt
    next_position_w[:, 2] = position_w[:, 2]
    next_velocity_w[:, 2] = 0.0
    return next_position_w, next_velocity_w


def test_piecewise_acceleration_current_segment_matches_constant_acceleration() -> None:
    segment_start_position_w = torch.tensor([[1.0, 2.0, 1.5], [-4.0, 0.5, 2.0]])
    segment_start_velocity_w = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, -3.0]])
    acceleration_w = torch.tensor([[0.2, -0.1, 1.0], [-0.05, 0.15, -1.0]])
    segment_step_count = torch.tensor([2, 5], dtype=torch.long)
    physics_dt = 0.1

    state = compute_piecewise_acceleration(
        segment_start_position_w,
        segment_start_velocity_w,
        acceleration_w,
        segment_step_count,
        physics_dt,
        fixed_height=True,
    )

    elapsed_time = segment_step_count.to(torch.float32).unsqueeze(-1) * physics_dt
    expected_position = segment_start_position_w + segment_start_velocity_w * elapsed_time
    expected_position += 0.5 * acceleration_w * elapsed_time.square()
    expected_velocity = segment_start_velocity_w + acceleration_w * elapsed_time
    expected_acceleration = acceleration_w.clone()
    expected_position[:, 2] = segment_start_position_w[:, 2]
    expected_velocity[:, 2] = 0.0
    expected_acceleration[:, 2] = 0.0
    torch.testing.assert_close(state.p_target_w, expected_position)
    torch.testing.assert_close(state.v_target_w, expected_velocity)
    torch.testing.assert_close(state.a_target_w, expected_acceleration)


def test_piecewise_manager_switches_segments_at_configured_duration() -> None:
    base_cfg = TargetMotionManagerCfg(max_acceleration=0.3)
    train_cfg = replace(
        base_cfg.train,
        mode_probabilities=(0.0, 0.0, 0.0, 1.0),
        piecewise_acceleration_x_range=(0.1, 0.1),
        piecewise_acceleration_y_range=(0.0, 0.0),
        piecewise_segment_duration_steps_range=(2, 2),
    )
    cfg = replace(base_cfg, train=train_cfg)
    manager = TargetMotionManager(num_envs=1, cfg=cfg, physics_dt=0.5, device="cpu")
    generator = make_split_generator(5, cfg, "train")
    p_initial_w = torch.tensor([[5.0, 0.0, 1.5]])
    v_initial_w = torch.tensor([[1.0, 0.0, 0.0]])

    manager.reset(torch.tensor([0]), p_initial_w, v_initial_w, generator, "train")
    state_after_one_step = manager.step()
    state_after_two_steps = manager.step()

    acceleration_w = torch.tensor([[0.1, 0.0, 0.0]])
    expected_step_one_position, expected_step_one_velocity = _independent_exact_accel_step(
        p_initial_w, v_initial_w, acceleration_w, 0.5
    )
    expected_boundary_position, expected_boundary_velocity = _independent_exact_accel_step(
        expected_step_one_position, expected_step_one_velocity, acceleration_w, 0.5
    )
    assert int(manager.segment_switch_count[0].item()) == 1
    assert int(manager.segment_step_count[0].item()) == 0
    torch.testing.assert_close(state_after_one_step.p_target_w, expected_step_one_position)
    torch.testing.assert_close(state_after_one_step.v_target_w, expected_step_one_velocity)
    torch.testing.assert_close(state_after_two_steps.p_target_w, expected_boundary_position)
    torch.testing.assert_close(state_after_two_steps.v_target_w, expected_boundary_velocity)
