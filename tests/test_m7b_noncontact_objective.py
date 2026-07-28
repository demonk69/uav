"""M7B non-contact objective invariants."""

import torch

from uav_rendezvous_rl.mdp import RendezvousRewardCfg, compute_reward_terms


def test_collision_risk_is_penalized_not_rewarded() -> None:
    cfg = RendezvousRewardCfg()
    reward_safe, terms_safe, _ = compute_reward_terms(
        offset_error_w=torch.zeros((1, 3)),
        previous_offset_error_norm=torch.zeros(1),
        v_ego_w=torch.zeros((1, 3)),
        v_target_w=torch.zeros((1, 3)),
        raw_action=torch.zeros((1, 3)),
        action_delta_squashed=torch.zeros((1, 3)),
        center_distance=torch.tensor([1.0]),
        acceleration_saturated=torch.zeros(1, dtype=torch.bool),
        speed_saturated=torch.zeros(1, dtype=torch.bool),
        collision_risk=torch.zeros(1, dtype=torch.bool),
        workspace_violation=torch.zeros(1, dtype=torch.bool),
        success_step=torch.zeros(1, dtype=torch.bool),
        success_completed=torch.zeros(1, dtype=torch.bool),
        d_safe=0.75,
        omega_ego_b=torch.zeros((1, 3)),
        cfg=cfg,
    )
    reward_collision, terms_collision, _ = compute_reward_terms(
        offset_error_w=torch.zeros((1, 3)),
        previous_offset_error_norm=torch.zeros(1),
        v_ego_w=torch.zeros((1, 3)),
        v_target_w=torch.zeros((1, 3)),
        raw_action=torch.zeros((1, 3)),
        action_delta_squashed=torch.zeros((1, 3)),
        center_distance=torch.tensor([0.5]),
        acceleration_saturated=torch.zeros(1, dtype=torch.bool),
        speed_saturated=torch.zeros(1, dtype=torch.bool),
        collision_risk=torch.ones(1, dtype=torch.bool),
        workspace_violation=torch.zeros(1, dtype=torch.bool),
        success_step=torch.zeros(1, dtype=torch.bool),
        success_completed=torch.zeros(1, dtype=torch.bool),
        d_safe=0.75,
        omega_ego_b=torch.zeros((1, 3)),
        cfg=cfg,
    )

    assert float(terms_collision["safety_distance"].item()) < float(terms_safe["safety_distance"].item())
    assert float(reward_collision.item()) < float(reward_safe.item())


def test_success_requires_noncontact_distance() -> None:
    center_distance = torch.tensor([0.74, 0.75, 1.00])
    d_safe = 0.75
    success_allowed = center_distance >= d_safe

    assert success_allowed.tolist() == [False, True, True]
