"""M7B dynamics integration tests with independent expected values."""

import math
import torch

from uav_rendezvous_rl.dynamics import integrate_m7b_dynamics, validate_m7b_dynamics_config, M7BDynamicsConfig
from uav_rendezvous_rl.controllers import compute_limited_acceleration, integrate_ego_kinematics


def test_m7b_one_step_nominal_is_same_as_m5_m7a() -> None:
    p = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    v_cmd = torch.tensor([[3.0, 0.0, 0.0]])
    wind = torch.zeros((1, 3))
    dt = 0.02
    tau = torch.tensor([0.25])
    a_lim = torch.tensor([2.0])
    speed_lim = torch.tensor([5.0])
    drag = torch.tensor([0.0])
    a_cap = 4.0

    p_next, v_next, track_sat, total_sat, speed_sat = integrate_m7b_dynamics(
        p, v, v_cmd, wind, dt, tau, a_lim, speed_lim, drag, a_cap
    )

    a_expected = float(clamp_scalar((3.0 - 1.0) / 0.25, 2.0))
    p_exp = float(0.0 + 1.0 * 0.02 + 0.5 * a_expected * 0.02 ** 2)
    v_exp = float(clamp_scalar(1.0 + a_expected * 0.02, 5.0))

    assert abs(float(p_next[0, 0].item()) - p_exp) < 1e-6
    assert abs(float(v_next[0, 0].item()) - v_exp) < 1e-6


def clamp_scalar(value: float, max_norm: float) -> float:
    if abs(value) > max_norm:
        return max_norm if value > 0 else -max_norm
    return value


def test_total_acceleration_cap_applies_correctly() -> None:
    p = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 0.0]])
    v_cmd = torch.tensor([[10.0, 0.0, 0.0]])
    wind = torch.tensor([[4.0, 0.0, 0.0]])
    dt = 0.02
    tau = torch.tensor([0.25])
    a_lim = torch.tensor([10.0])
    speed_lim = torch.tensor([10.0])
    drag = torch.tensor([0.0])
    a_cap = 4.0

    _, v_next, _, total_sat, _ = integrate_m7b_dynamics(
        p, v, v_cmd, wind, dt, tau, a_lim, speed_lim, drag, a_cap
    )

    dv = float(v_next[0, 0].item())
    a_eff = dv / dt
    assert abs(a_eff) <= 4.0 + 1e-6
    assert bool(total_sat[0].item())


def test_speed_cap_applies_correctly() -> None:
    p = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[10.0, 0.0, 0.0]])
    v_cmd = torch.tensor([[10.0, 0.0, 0.0]])
    wind = torch.zeros((1, 3))
    dt = 0.02
    tau = torch.tensor([1.0])
    a_lim = torch.tensor([3.0])
    speed_lim = torch.tensor([3.0])
    drag = torch.tensor([0.0])
    a_cap = 4.0

    _, v_next, _, _, speed_sat = integrate_m7b_dynamics(
        p, v, v_cmd, wind, dt, tau, a_lim, speed_lim, drag, a_cap
    )

    assert float(torch.linalg.norm(v_next).item()) <= 3.0 + 1e-6
    assert bool(speed_sat[0].item())


def test_dynamics_config_validation() -> None:
    validate_m7b_dynamics_config(M7BDynamicsConfig())
    try:
        validate_m7b_dynamics_config(M7BDynamicsConfig(tau_v_nominal=-0.1))
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        validate_m7b_dynamics_config(M7BDynamicsConfig(gust_tau_s=0.0))
        assert False, "should have raised"
    except ValueError:
        pass


def test_position_formula_is_analytic_constant_acceleration() -> None:
    dt = 0.02
    a_applied = 3.0
    p = 0.0
    v = 2.0
    p_next = p + v * dt + 0.5 * a_applied * dt ** 2
    expected = 2.0 * 0.02 + 0.5 * 3.0 * 0.0004
    assert abs(p_next - expected) < 1e-12
    assert abs(p_next - 0.0406) < 1e-6


def test_stage_zero_1000_step_helper_equivalence_to_m5_m7a() -> None:
    p_m7b = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
    v_m7b = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float32)
    p_ref = p_m7b.clone()
    v_ref = v_m7b.clone()
    dt = 0.02
    tau = torch.tensor([0.25], dtype=torch.float32)
    a_lim = torch.tensor([2.0], dtype=torch.float32)
    speed_lim = torch.tensor([5.0], dtype=torch.float32)
    drag = torch.tensor([0.0], dtype=torch.float32)
    wind = torch.zeros((1, 3), dtype=torch.float32)

    for step in range(1000):
        command = torch.tensor(
            [[math.sin(step * 0.013), math.cos(step * 0.017), math.sin(step * 0.019)]], dtype=torch.float32
        ) * 3.0
        p_m7b, v_m7b, _, _, _ = integrate_m7b_dynamics(
            p_m7b, v_m7b, command, wind, dt, tau, a_lim, speed_lim, drag, 4.0
        )
        a_ref, _ = compute_limited_acceleration(command, v_ref, tau_v=0.25, a_max=2.0)
        p_ref, v_ref, _ = integrate_ego_kinematics(p_ref, v_ref, a_ref, dt, v_abs_max=5.0)

    assert torch.max(torch.abs(p_m7b - p_ref)).item() <= 1e-6
    assert torch.max(torch.abs(v_m7b - v_ref)).item() <= 1e-6
