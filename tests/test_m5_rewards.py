"""Pure PyTorch tests for M5 reward terms."""

import torch

from uav_rendezvous_rl.mdp import RendezvousRewardCfg, compute_reward_terms


def _reward(offset_error_value: float) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    cfg = RendezvousRewardCfg()
    offset_error_w = torch.tensor([[offset_error_value, 0.0, 0.0]])
    return compute_reward_terms(
        offset_error_w=offset_error_w,
        previous_offset_error_norm=torch.tensor([2.0]),
        v_ego_w=torch.zeros((1, 3)),
        v_target_w=torch.zeros((1, 3)),
        raw_action=torch.zeros((1, 3)),
        action_delta_squashed=torch.zeros((1, 3)),
        center_distance=torch.tensor([5.0]),
        acceleration_saturated=torch.tensor([False]),
        speed_saturated=torch.tensor([False]),
        collision_risk=torch.tensor([False]),
        workspace_violation=torch.tensor([False]),
        success_step=torch.tensor([False]),
        success_completed=torch.tensor([False]),
        d_safe=0.75,
        omega_ego_b=torch.zeros((1, 3)),
        cfg=cfg,
    )


def test_offset_reward_is_higher_for_smaller_offset_error() -> None:
    close_reward, close_terms, _ = _reward(0.2)
    far_reward, far_terms, _ = _reward(2.0)

    assert close_terms["offset"].item() > far_terms["offset"].item()
    assert close_reward.item() > far_reward.item()


def test_reward_terms_are_finite_and_collision_penalty_is_negative() -> None:
    cfg = RendezvousRewardCfg()
    reward, terms, offset_norm = compute_reward_terms(
        offset_error_w=torch.zeros((2, 3)),
        previous_offset_error_norm=torch.ones(2),
        v_ego_w=torch.zeros((2, 3)),
        v_target_w=torch.zeros((2, 3)),
        raw_action=torch.ones((2, 3)),
        action_delta_squashed=torch.ones((2, 3)) * 0.5,
        center_distance=torch.tensor([5.0, 0.5]),
        acceleration_saturated=torch.tensor([False, True]),
        speed_saturated=torch.tensor([False, True]),
        collision_risk=torch.tensor([False, True]),
        workspace_violation=torch.tensor([False, True]),
        success_step=torch.tensor([True, False]),
        success_completed=torch.tensor([False, False]),
        d_safe=0.75,
        omega_ego_b=torch.zeros((2, 3)),
        cfg=cfg,
    )

    assert bool(torch.isfinite(reward).all().item())
    assert bool(torch.isfinite(offset_norm).all().item())
    assert all(bool(torch.isfinite(value).all().item()) for value in terms.values())
    assert terms["safety_distance"][1].item() < terms["safety_distance"][0].item()
