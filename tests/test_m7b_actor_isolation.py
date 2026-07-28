"""M7B Actor isolation tests."""

import pathlib

import torch

from uav_rendezvous_rl.mdp import assemble_actor_observation


ROOT = pathlib.Path(__file__).resolve().parents[1]
M7B_ENV_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7b_env.py"


def test_actor_observation_remains_25d_and_uses_previous_issued_action() -> None:
    n = 2
    previous_issued = torch.ones((n, 3)) * 0.25
    actor = assemble_actor_observation(
        torch.zeros((n, 3)),
        torch.zeros((n, 3)),
        torch.zeros((n, 3)),
        torch.zeros((n, 6)),
        torch.zeros((n, 3)),
        previous_issued,
        torch.ones((n, 3)),
        5.0,
    )

    assert actor.shape == (n, 25)
    assert torch.allclose(actor[:, 18:21], previous_issued)


def test_m7b_env_does_not_feed_executed_action_to_actor() -> None:
    source = M7B_ENV_SOURCE.read_text(encoding="utf-8")

    actor_call = source.split("actor_obs = assemble_actor_observation", maxsplit=1)[1].split("mode_one_hot", maxsplit=1)[0]
    assert "previous_squashed_action" in actor_call
    assert "executed_squashed_action" not in actor_call


def test_actor_forbidden_privileged_terms_not_in_actor_assembly_slice() -> None:
    source = M7B_ENV_SOURCE.read_text(encoding="utf-8")
    actor_call = source.split("actor_obs = assemble_actor_observation", maxsplit=1)[1].split("mode_one_hot", maxsplit=1)[0]

    for forbidden in ("_m7b_tau_velocity_scale", "_m7b_action_delay_steps", "_cached_wind_accel_w", "gust_remaining"):
        assert forbidden not in actor_call
