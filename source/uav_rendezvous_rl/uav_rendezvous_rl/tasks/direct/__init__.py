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
