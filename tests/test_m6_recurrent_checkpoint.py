"""Checkpoint smoke test for the M6 recurrent policy module."""

import torch
from rsl_rl.modules import ActorCriticRecurrent
from tensordict import TensorDict


def test_recurrent_policy_state_dict_round_trips(tmp_path) -> None:
    obs = TensorDict(
        {
            "policy": torch.zeros((2, 25), dtype=torch.float32),
            "critic": torch.zeros((2, 57), dtype=torch.float32),
        },
        batch_size=[2],
    )
    kwargs = dict(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=3,
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    policy = ActorCriticRecurrent(**kwargs)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": {}, "iter": 7, "infos": None}, checkpoint)

    restored = ActorCriticRecurrent(**kwargs)
    loaded = torch.load(checkpoint, weights_only=False)
    assert restored.load_state_dict(loaded["model_state_dict"])
    assert loaded["iter"] == 7

    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
