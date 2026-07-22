"""Gymnasium task registration tests."""

import gymnasium as gym


TASK_ID = "Isaac-Uav-Rendezvous-Direct-v0"
BASELINE_TASK_ID = "Isaac-Uav-Rendezvous-Baseline-v0"


def test_task_registers() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(TASK_ID)

    assert spec.id == TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_env:UavRendezvousEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_env_cfg:UavRendezvousEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.agents.rsl_rl_ppo_cfg:UavRendezvousPPORunnerCfg"
    )


def test_baseline_task_registers_independently() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(BASELINE_TASK_ID)

    assert spec.id == BASELINE_TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_baseline_env:UavRendezvousBaselineEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_baseline_env_cfg:UavRendezvousBaselineEnvCfg"
    )
