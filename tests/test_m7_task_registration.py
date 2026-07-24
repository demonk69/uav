"""M7A task registration tests."""

import gymnasium as gym


M7A_GRU_TASK_ID = "Isaac-Uav-Rendezvous-M7A-GRU-v0"
M7A_FEEDFORWARD_TASK_ID = "Isaac-Uav-Rendezvous-M7A-Feedforward-v0"


def test_m7a_gru_task_registers_independently() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(M7A_GRU_TASK_ID)

    assert spec.id == M7A_GRU_TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7a_env:UavRendezvousM7AEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7a_env_cfg:UavRendezvousM7AEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.agents.rsl_rl_ppo_cfg:UavRendezvousM7AGRUPPORunnerCfg"
    )


def test_m7a_feedforward_task_uses_same_environment() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    spec = gym.spec(M7A_FEEDFORWARD_TASK_ID)

    assert spec.id == M7A_FEEDFORWARD_TASK_ID
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7a_env:UavRendezvousM7AEnv"
    assert spec.kwargs["env_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7a_env_cfg:UavRendezvousM7AEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == (
        "uav_rendezvous_rl.tasks.direct.agents.rsl_rl_ppo_cfg:UavRendezvousM7AFeedforwardPPORunnerCfg"
    )


def test_m2_through_m6_task_registration_entry_points_are_unchanged() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401

    expected = {
        "Isaac-Uav-Rendezvous-Direct-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_env:UavRendezvousEnv",
        "Isaac-Uav-Rendezvous-Baseline-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_baseline_env:UavRendezvousBaselineEnv",
        "Isaac-Uav-Rendezvous-RL-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_rl_env:UavRendezvousRLEnv",
        "Isaac-Uav-Rendezvous-Recurrent-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
        "Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
    }
    for task_id, entry_point in expected.items():
        assert gym.spec(task_id).entry_point == entry_point
