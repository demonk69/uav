"""Static tests that M6 keeps forbidden target-motion information out of Actor observations."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
M5_ENV_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_rl_env.py"
M6_ENV_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_recurrent_env.py"
M6_CFG_SOURCE = ROOT / "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_recurrent_env_cfg.py"


def test_m6_reuses_m5_actor_observation_assembly_without_new_actor_inputs() -> None:
    m6_text = M6_ENV_SOURCE.read_text(encoding="utf-8")
    m5_text = M5_ENV_SOURCE.read_text(encoding="utf-8")

    assert "class UavRendezvousRecurrentEnv(UavRendezvousRLEnv)" in m6_text
    assert "assemble_actor_observation" not in m6_text
    assert "mode_one_hot" not in m6_text
    assert "target_motion_current_params" not in m6_text
    assert "assemble_actor_observation(" in m5_text
    assert "mode_one_hot =" in m5_text
    assert m5_text.index("actor_obs = assemble_actor_observation") < m5_text.index("mode_one_hot =")


def test_m6_cfg_keeps_fixed_25d_actor_and_57d_critic_contract_by_inheritance() -> None:
    text = M6_CFG_SOURCE.read_text(encoding="utf-8")

    assert "UavRendezvousRecurrentEnvCfg(UavRendezvousRLEnvCfg)" in text
    assert "observation_space" not in text
    assert "state_space" not in text
