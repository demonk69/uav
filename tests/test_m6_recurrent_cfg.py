"""Static tests for the M6 recurrent PPO configuration."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py"
ENV_CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_recurrent_env_cfg.py"


def _class_block(text: str, class_name: str, next_class_name: str | None = None) -> str:
    start = text.index(f"class {class_name}")
    if next_class_name is None:
        return text[start:]
    end = text.index(f"class {next_class_name}", start)
    return text[start:end]


def test_m6_gru_config_matches_required_contract() -> None:
    text = CFG_SOURCE.read_text(encoding="utf-8")
    block = _class_block(text, "UavRendezvousRecurrentPPORunnerCfg", "UavRendezvousRLPPORunnerCfg")

    assert "RslRlPpoActorCriticRecurrentCfg" in block
    assert "num_steps_per_env = 128" in block
    assert "experiment_name = \"uav_rendezvous_m6_gru\"" in block
    assert "clip_actions = None" in block
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in block
    assert "init_noise_std=0.5" in block
    assert "actor_obs_normalization=True" in block
    assert "critic_obs_normalization=True" in block
    assert "actor_hidden_dims=[128, 128]" in block
    assert "critic_hidden_dims=[128, 128]" in block
    assert 'rnn_type="gru"' in block
    assert "rnn_hidden_dim=128" in block
    assert "rnn_num_layers=1" in block


def test_m6_feedforward_ablation_is_not_recurrent_and_uses_same_rollout_length() -> None:
    text = CFG_SOURCE.read_text(encoding="utf-8")
    block = _class_block(text, "UavRendezvousM6FeedforwardAblationPPORunnerCfg")

    assert "RslRlPpoActorCriticRecurrentCfg" not in block
    assert "RslRlPpoActorCriticCfg" in block
    assert "num_steps_per_env = 128" in block
    assert "experiment_name = \"uav_rendezvous_m6_feedforward_ablation\"" in block
    assert "clip_actions = None" in block
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in block


def test_m6_env_cfg_uses_explicit_four_mode_mixed_distribution() -> None:
    text = ENV_CFG_SOURCE.read_text(encoding="utf-8")

    assert "mixed = (0.25, 0.25, 0.25, 0.25)" in text
    assert "force_mode_cycle_on_reset=False" in text
    assert "UavRendezvousRecurrentEnvCfg(UavRendezvousRLEnvCfg)" in text
