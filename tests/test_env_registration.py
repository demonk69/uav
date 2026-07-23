"""Gymnasium task registration tests."""

import gymnasium as gym


TASK_ID = "Isaac-Uav-Rendezvous-Direct-v0"
BASELINE_TASK_ID = "Isaac-Uav-Rendezvous-Baseline-v0"
RL_TASK_ID = "Isaac-Uav-Rendezvous-RL-v0"
RECURRENT_TASK_ID = "Isaac-Uav-Rendezvous-Recurrent-v0"
M6_FEEDFORWARD_ABLATION_TASK_ID = "Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0"


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


def test_rl_task_registers_independently() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(RL_TASK_ID)

    assert spec.id == RL_TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_rl_env:UavRendezvousRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_rl_env_cfg:UavRendezvousRLEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.agents.rsl_rl_ppo_cfg:UavRendezvousRLPPORunnerCfg"
    )


def test_recurrent_task_registers_independently() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(RECURRENT_TASK_ID)

    assert spec.id == RECURRENT_TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env_cfg:UavRendezvousRecurrentEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.agents.rsl_rl_ppo_cfg:UavRendezvousRecurrentPPORunnerCfg"
    )


def test_m6_feedforward_ablation_task_uses_same_environment() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(M6_FEEDFORWARD_ABLATION_TASK_ID)

    assert spec.id == M6_FEEDFORWARD_ABLATION_TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env_cfg:UavRendezvousRecurrentEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.agents.rsl_rl_ppo_cfg:UavRendezvousM6FeedforwardAblationPPORunnerCfg"
    )
