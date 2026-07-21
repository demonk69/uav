"""Direct workflow task registration for M1."""

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
