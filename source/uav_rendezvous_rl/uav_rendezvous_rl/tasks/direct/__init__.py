"""Direct workflow task registrations."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Uav-Rendezvous-Direct-v0",
    entry_point=f"{__name__}.uav_rendezvous_env:UavRendezvousEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_env_cfg:UavRendezvousEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Uav-Rendezvous-Baseline-v0",
    entry_point=f"{__name__}.uav_rendezvous_baseline_env:UavRendezvousBaselineEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_baseline_env_cfg:UavRendezvousBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Uav-Rendezvous-RL-v0",
    entry_point=f"{__name__}.uav_rendezvous_rl_env:UavRendezvousRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_rl_env_cfg:UavRendezvousRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousRLPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Uav-Rendezvous-Recurrent-v0",
    entry_point=f"{__name__}.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_recurrent_env_cfg:UavRendezvousRecurrentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0",
    entry_point=f"{__name__}.uav_rendezvous_recurrent_env:UavRendezvousRecurrentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_recurrent_env_cfg:UavRendezvousRecurrentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousM6FeedforwardAblationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Uav-Rendezvous-M7A-GRU-v0",
    entry_point=f"{__name__}.uav_rendezvous_m7a_env:UavRendezvousM7AEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_m7a_env_cfg:UavRendezvousM7AEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousM7AGRUPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Uav-Rendezvous-M7A-Feedforward-v0",
    entry_point=f"{__name__}.uav_rendezvous_m7a_env:UavRendezvousM7AEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_rendezvous_m7a_env_cfg:UavRendezvousM7AEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UavRendezvousM7AFeedforwardPPORunnerCfg",
    },
)
