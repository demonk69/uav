"""Configuration for independent M7A partial-observability rendezvous tasks."""

from __future__ import annotations

from isaaclab.utils import configclass
from uav_rendezvous_rl.observations import ObservationPipelineCfg, make_m7a_observation_cfg

from .uav_rendezvous_recurrent_env_cfg import UavRendezvousRecurrentEnvCfg


@configclass
class UavRendezvousM7AEnvCfg(UavRendezvousRecurrentEnvCfg):
    """M7A task config with controlled causal observation degradation."""

    # Stage 0 is the clean regression default. Training/evaluation scripts may override this by stage.
    observation_degradation: ObservationPipelineCfg = make_m7a_observation_cfg(0)
