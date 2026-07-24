"""Static M7A Actor/Critic information-boundary tests."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
M7_ENV_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env.py"
M7_CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env_cfg.py"


def _method_block(text: str, method_name: str, next_method_name: str) -> str:
    start = text.index(f"    def {method_name}")
    end = text.index(f"    def {next_method_name}", start)
    return text[start:end]


def test_actor_uses_only_degraded_relative_observations() -> None:
    text = M7_ENV_SOURCE.read_text(encoding="utf-8")
    block = _method_block(text, "_get_observations", "get_m7a_diagnostics")
    actor_block = block[block.index("actor_obs = assemble_actor_observation") : block.index("mode_one_hot =")]

    assert "self.p_rel_obs_w" in actor_block
    assert "self.v_rel_obs_w" in actor_block
    assert "self.p_rel_w," not in actor_block
    assert "self.v_rel_w," not in actor_block
    assert "a_target_w" not in actor_block
    assert "mode_id" not in actor_block
    assert "mode_one_hot" not in actor_block
    assert "target_motion_current_params" not in actor_block
    assert "dropout" not in actor_block
    assert "age" not in actor_block
    assert "history" not in actor_block


def test_critic_is_assembled_after_actor_and_remains_separate() -> None:
    text = M7_ENV_SOURCE.read_text(encoding="utf-8")
    block = _method_block(text, "_get_observations", "get_m7a_diagnostics")

    assert block.index("actor_obs = assemble_actor_observation") < block.index("critic_obs = assemble_critic_observation")
    assert "mode_one_hot =" in block
    assert "target_motion_current_params =" in block
    assert "return {\"policy\": actor_obs, \"critic\": critic_obs}" in block


def test_m7a_cfg_keeps_25d_actor_and_57d_critic_by_inheritance() -> None:
    text = M7_CFG_SOURCE.read_text(encoding="utf-8")

    assert "UavRendezvousM7AEnvCfg(UavRendezvousRecurrentEnvCfg)" in text
    assert "observation_space" not in text
    assert "state_space" not in text
    assert "observation_degradation" in text
