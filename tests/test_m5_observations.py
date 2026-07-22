"""Pure PyTorch tests for the fixed M5 Actor and Critic observation layouts."""

import torch
import torch.nn.functional as F

from uav_rendezvous_rl.mdp import (
    assemble_actor_observation,
    assemble_critic_observation,
    encode_target_motion_current_params,
    identity_rotation_6d,
)
from uav_rendezvous_rl.motions.configs import MODE_CONSTANT_VELOCITY


def test_actor_observation_has_fixed_25d_order_and_previous_squashed_action() -> None:
    num_envs = 2
    p_rel_w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v_rel_w = torch.ones((num_envs, 3)) * 2.0
    v_ego_w = torch.ones((num_envs, 3)) * 3.0
    r_ego_6d = identity_rotation_6d(num_envs, "cpu")
    omega_ego_b = torch.zeros((num_envs, 3))
    previous_squashed_action = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]])
    b_des_w = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

    obs = assemble_actor_observation(
        p_rel_w, v_rel_w, v_ego_w, r_ego_6d, omega_ego_b, previous_squashed_action, b_des_w, 5.0
    )

    assert obs.shape == (num_envs, 25)
    torch.testing.assert_close(obs[:, 0:3], p_rel_w)
    torch.testing.assert_close(obs[:, 3:6], v_rel_w)
    torch.testing.assert_close(obs[:, 6:9], v_ego_w)
    torch.testing.assert_close(obs[:, 9:15], r_ego_6d)
    torch.testing.assert_close(obs[:, 15:18], omega_ego_b)
    torch.testing.assert_close(obs[:, 18:21], previous_squashed_action)
    torch.testing.assert_close(obs[:, 21:24], b_des_w)
    torch.testing.assert_close(obs[:, 24], torch.full((num_envs,), 5.0))


def test_critic_observation_has_fixed_57d_actor_prefix_and_privileged_tail() -> None:
    num_envs = 2
    actor_obs = torch.arange(num_envs * 25, dtype=torch.float32).reshape(num_envs, 25)
    p_ego_w = torch.ones((num_envs, 3))
    p_target_w = torch.ones((num_envs, 3)) * 2.0
    v_target_w = torch.ones((num_envs, 3)) * 3.0
    a_target_w = torch.ones((num_envs, 3)) * 4.0
    r_target_6d = identity_rotation_6d(num_envs, "cpu")
    omega_target_b = torch.zeros((num_envs, 3))
    mode_id = torch.full((num_envs,), MODE_CONSTANT_VELOCITY, dtype=torch.long)
    mode_one_hot = F.one_hot(mode_id, num_classes=4).to(torch.float32)
    params = encode_target_motion_current_params(
        mode_id,
        v0_w=torch.ones((num_envs, 3)),
        constant_acceleration_w=torch.zeros((num_envs, 3)),
        turn_omega=torch.zeros(num_envs),
        current_acceleration_w=torch.zeros((num_envs, 3)),
        max_speed=8.0,
        max_acceleration=0.25,
        max_turn_omega=0.1,
    )
    episode_phase = torch.tensor([0.0, 0.5])

    critic_obs = assemble_critic_observation(
        actor_obs,
        p_ego_w,
        p_target_w,
        v_target_w,
        a_target_w,
        r_target_6d,
        omega_target_b,
        mode_one_hot,
        params,
        episode_phase,
    )

    assert critic_obs.shape == (num_envs, 57)
    torch.testing.assert_close(critic_obs[:, 0:25], actor_obs)
    torch.testing.assert_close(critic_obs[:, 25:28], p_ego_w)
    torch.testing.assert_close(critic_obs[:, 28:31], p_target_w)
    torch.testing.assert_close(critic_obs[:, 31:34], v_target_w)
    torch.testing.assert_close(critic_obs[:, 34:37], a_target_w)
    torch.testing.assert_close(critic_obs[:, 37:43], r_target_6d)
    torch.testing.assert_close(critic_obs[:, 43:46], omega_target_b)
    torch.testing.assert_close(critic_obs[:, 46:50], mode_one_hot)
    torch.testing.assert_close(critic_obs[:, 50:56], params)
    torch.testing.assert_close(critic_obs[:, 56], episode_phase)
