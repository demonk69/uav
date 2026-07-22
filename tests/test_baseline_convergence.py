"""Pure PyTorch convergence test for M4 ConstantVelocity baseline."""

import torch

from uav_rendezvous_rl.controllers import (
    compute_baseline_velocity_command,
    compute_limited_acceleration,
    integrate_ego_kinematics,
)


def test_constant_velocity_scenario_converges_to_offset_and_velocity_match() -> None:
    p_target_w = torch.tensor([[4.0, 0.5, 1.5]])
    v_target_w = torch.tensor([[0.7, -0.1, 0.0]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0]])
    p_ego_w = p_target_w + b_des_w + torch.tensor([[4.0, -1.0, 0.0]])
    v_ego_w = torch.zeros((1, 3))
    dt = 0.01
    success_steps = 0

    for _ in range(2000):
        v_cmd_w, *_ = compute_baseline_velocity_command(p_ego_w, p_target_w, v_target_w, b_des_w, 0.5, 3.0)
        a_cmd_w, _ = compute_limited_acceleration(v_cmd_w, v_ego_w, 0.25, 2.0)
        p_ego_w, v_ego_w, _ = integrate_ego_kinematics(p_ego_w, v_ego_w, a_cmd_w, dt, 5.0)
        p_target_w = p_target_w + v_target_w * dt
        offset_error = torch.linalg.norm(p_ego_w - p_target_w - b_des_w, dim=1)
        relative_speed = torch.linalg.norm(v_ego_w - v_target_w, dim=1)
        safe_distance = torch.linalg.norm(p_target_w - p_ego_w, dim=1) >= 0.75
        if bool(((offset_error < 0.5) & (relative_speed < 0.3) & safe_distance).all().item()):
            success_steps += 1
        else:
            success_steps = 0

    assert success_steps >= 100
    assert bool(torch.isfinite(p_ego_w).all().item())
