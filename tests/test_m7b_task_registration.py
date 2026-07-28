"""M7B task registration and M2-M7A preservation tests."""

import gymnasium as gym


M7B_FF_TASK = "Isaac-Uav-Rendezvous-M7B-Feedforward-v0"
M7B_GRU_TASK = "Isaac-Uav-Rendezvous-M7B-GRU-v0"


def test_m7b_feedforward_registration() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401
    spec = gym.spec(M7B_FF_TASK)
    assert spec.id == M7B_FF_TASK
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7b_env:UavRendezvousM7BEnv"
    assert "uav_rendezvous_m7b_env_cfg:UavRendezvousM7BEnvCfg" in spec.kwargs["env_cfg_entry_point"]


def test_m7b_gru_registration() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401
    spec = gym.spec(M7B_GRU_TASK)
    assert spec.id == M7B_GRU_TASK
    assert spec.entry_point == "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7b_env:UavRendezvousM7BEnv"


def test_m2_through_m7a_tasks_unchanged() -> None:
    import uav_rendezvous_rl.tasks  # noqa: F401
    expected = {
        "Isaac-Uav-Rendezvous-Direct-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_env:UavRendezvousEnv",
        "Isaac-Uav-Rendezvous-Baseline-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_baseline_env:UavRendezvousBaselineEnv",
        "Isaac-Uav-Rendezvous-RL-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_rl_env:UavRendezvousRLEnv",
        "Isaac-Uav-Rendezvous-Recurrent-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
        "Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
        "Isaac-Uav-Rendezvous-M7A-GRU-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7a_env:UavRendezvousM7AEnv",
        "Isaac-Uav-Rendezvous-M7A-Feedforward-v0": "uav_rendezvous_rl.tasks.direct.uav_rendezvous_m7a_env:UavRendezvousM7AEnv",
    }
    for task_id, entry_point in expected.items():
        assert gym.spec(task_id).entry_point == entry_point
