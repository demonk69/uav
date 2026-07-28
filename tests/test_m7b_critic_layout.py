"""M7B 65D critic layout tests with exact slice checks."""

import torch
from pathlib import Path

from uav_rendezvous_rl.mdp import assemble_critic_observation_m7b


ROOT = Path(__file__).parents[1]
M7B_CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7b_env_cfg.py"


def _zeros(shape: tuple) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def test_critic_65d_slice_values() -> None:
    n = 2
    actor = torch.ones((n, 25)) * 1.0
    p_ego = torch.ones((n, 3)) * 2.0
    p_tgt = torch.ones((n, 3)) * 3.0
    v_tgt = torch.ones((n, 3)) * 4.0
    a_tgt = torch.ones((n, 3)) * 5.0
    r_tgt = torch.ones((n, 6)) * 6.0
    omega = torch.ones((n, 3)) * 7.0
    mode = torch.ones((n, 4)) * 8.0
    params = torch.ones((n, 6)) * 9.0
    phase = torch.ones(n) * 10.0
    tau = torch.ones(n) * 11.0
    acc = torch.ones(n) * 12.0
    spd = torch.ones(n) * 13.0
    drag = torch.ones(n) * 14.0
    norm_delay = torch.ones(n) * 15.0
    wind = torch.ones((n, 3)) * 16.0

    result = assemble_critic_observation_m7b(
        actor, p_ego, p_tgt, v_tgt, a_tgt, r_tgt, omega, mode, params, phase,
        tau, acc, spd, drag, norm_delay, wind,
    )

    assert result.shape == (n, 65)
    assert torch.allclose(result[0, 0:25], actor[0])
    assert torch.allclose(result[0, 25:28], p_ego[0])
    assert torch.allclose(result[0, 28:31], p_tgt[0])
    assert torch.allclose(result[0, 31:34], v_tgt[0])
    assert torch.allclose(result[0, 34:37], a_tgt[0])
    assert torch.allclose(result[0, 37:43], r_tgt[0])
    assert torch.allclose(result[0, 43:46], omega[0])
    assert torch.allclose(result[0, 46:50], mode[0])
    assert torch.allclose(result[0, 50:56], params[0])
    assert abs(float(result[0, 56].item()) - 10.0) < 1e-6
    assert abs(float(result[0, 57].item()) - 11.0) < 1e-6
    assert abs(float(result[0, 58].item()) - 12.0) < 1e-6
    assert abs(float(result[0, 59].item()) - 13.0) < 1e-6
    assert abs(float(result[0, 60].item()) - 14.0) < 1e-6
    assert abs(float(result[0, 61].item()) - 15.0) < 1e-6
    assert torch.allclose(result[0, 62:65], torch.tensor([16.0, 16.0, 16.0]))


def test_m7a_57d_assembly_unchanged() -> None:
    from uav_rendezvous_rl.mdp import assemble_critic_observation
    n = 2
    actor = torch.ones((n, 25)) * 1.0
    p_ego = torch.ones((n, 3)) * 2.0
    p_tgt = torch.ones((n, 3)) * 3.0
    v_tgt = torch.ones((n, 3)) * 4.0
    a_tgt = torch.ones((n, 3)) * 5.0
    r_tgt = torch.ones((n, 6)) * 6.0
    omega = torch.ones((n, 3)) * 7.0
    mode = torch.ones((n, 4)) * 8.0
    params = torch.ones((n, 6)) * 9.0
    phase = torch.ones(n) * 10.0

    result = assemble_critic_observation(actor, p_ego, p_tgt, v_tgt, a_tgt, r_tgt, omega, mode, params, phase)

    assert result.shape == (n, 57)


def test_m7b_env_cfg_declares_65d_state_space() -> None:
    text = M7B_CFG_SOURCE.read_text(encoding="utf-8")

    assert "state_space = 65" in text
