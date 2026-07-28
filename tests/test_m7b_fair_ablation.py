"""M7B feedforward/GRU fair-ablation contract tests."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/agents/rsl_rl_ppo_cfg.py"


def _class_block(text: str, class_name: str, next_class_name: str | None = None) -> str:
    start = text.index(f"class {class_name}")
    if next_class_name is None:
        return text[start:]
    end = text.index(f"class {next_class_name}", start)
    return text[start:end]


def test_m7b_feedforward_and_gru_use_matched_rollout_budget() -> None:
    text = CFG_SOURCE.read_text(encoding="utf-8")
    ff = _class_block(text, "UavRendezvousM7BFeedforwardPPORunnerCfg", "UavRendezvousM7BGRUPPORunnerCfg")
    gru = _class_block(text, "UavRendezvousM7BGRUPPORunnerCfg", "UavRendezvousRecurrentPPORunnerCfg")

    assert "num_steps_per_env = 128" in ff
    assert "num_steps_per_env = 128" in gru
    assert "max_iterations = 300" in ff
    assert "max_iterations = 300" in gru
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in ff
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in gru


def test_m7b_gru_is_recurrent_secondary_ablation() -> None:
    text = CFG_SOURCE.read_text(encoding="utf-8")
    ff = _class_block(text, "UavRendezvousM7BFeedforwardPPORunnerCfg", "UavRendezvousM7BGRUPPORunnerCfg")
    gru = _class_block(text, "UavRendezvousM7BGRUPPORunnerCfg", "UavRendezvousRecurrentPPORunnerCfg")

    assert "RslRlPpoActorCriticCfg" in ff
    assert "RslRlPpoActorCriticRecurrentCfg" not in ff
    assert "RslRlPpoActorCriticRecurrentCfg" in gru
    assert 'rnn_type="gru"' in gru
