"""Static tests for the M5 feedforward PPO configuration."""

from pathlib import Path


def test_m5_ppo_config_is_feedforward_asymmetric_and_unclipped() -> None:
    source = Path(__file__).parents[1] / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py"
    text = source.read_text(encoding="utf-8")

    class_start = text.index("class UavRendezvousRLPPORunnerCfg")
    class_end = text.index("class UavRendezvousM6FeedforwardAblationPPORunnerCfg", class_start)
    m5_class = text[class_start:class_end]

    assert "RslRlPpoActorCriticRecurrentCfg" not in m5_class
    assert "clip_actions = None" in m5_class
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in m5_class
    assert "num_steps_per_env = 64" in m5_class
    assert "init_noise_std=0.5" in m5_class
    assert "actor_hidden_dims=[128, 128]" in m5_class
    assert "critic_hidden_dims=[128, 128]" in m5_class
