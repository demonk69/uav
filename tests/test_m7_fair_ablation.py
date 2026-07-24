"""M7A GRU/feedforward fair-ablation static checks."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py"
TASK_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/__init__.py"
ENV_CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env_cfg.py"


def _class_block(text: str, class_name: str, next_class_name: str | None = None) -> str:
    start = text.index(f"class {class_name}")
    if next_class_name is None:
        return text[start:]
    return text[start : text.index(f"class {next_class_name}", start)]


def test_m7a_gru_and_feedforward_match_except_policy_type_and_experiment_name() -> None:
    text = CFG_SOURCE.read_text(encoding="utf-8")
    gru = _class_block(text, "UavRendezvousM7AGRUPPORunnerCfg", "UavRendezvousM7AFeedforwardPPORunnerCfg")
    ff = _class_block(text, "UavRendezvousM7AFeedforwardPPORunnerCfg")

    shared_fragments = [
        "num_steps_per_env = 128",
        "max_iterations = 100",
        "save_interval = 25",
        "clip_actions = None",
        'obs_groups = {"policy": ["policy"], "critic": ["critic"]}',
        "init_noise_std=0.5",
        "actor_obs_normalization=True",
        "critic_obs_normalization=True",
        "actor_hidden_dims=[128, 128]",
        "critic_hidden_dims=[128, 128]",
        'activation="elu"',
        "value_loss_coef=1.0",
        "clip_param=0.2",
        "entropy_coef=0.005",
        "num_learning_epochs=4",
        "num_mini_batches=4",
        "learning_rate=3.0e-4",
        'schedule="adaptive"',
        "gamma=0.99",
        "lam=0.95",
        "desired_kl=0.01",
        "max_grad_norm=1.0",
    ]
    for fragment in shared_fragments:
        assert fragment in gru
        assert fragment in ff
    assert "RslRlPpoActorCriticRecurrentCfg" in gru
    assert "RslRlPpoActorCriticRecurrentCfg" not in ff
    assert "RslRlPpoActorCriticCfg" in ff
    assert 'rnn_type="gru"' in gru


def test_m7a_tasks_share_identical_environment_config() -> None:
    text = TASK_SOURCE.read_text(encoding="utf-8")
    assert text.count("uav_rendezvous_m7a_env:UavRendezvousM7AEnv") == 2
    assert text.count("uav_rendezvous_m7a_env_cfg:UavRendezvousM7AEnvCfg") == 2


def test_m7a_default_env_cfg_is_clean_stage_zero() -> None:
    text = ENV_CFG_SOURCE.read_text(encoding="utf-8")
    assert "make_m7a_observation_cfg(0)" in text
    assert "UavRendezvousM7AEnvCfg(UavRendezvousRecurrentEnvCfg)" in text
