"""Configuration for independent M7B dynamics-robustness rendezvous tasks."""

from __future__ import annotations

from isaaclab.utils import configclass
from uav_rendezvous_rl.dynamics import M7BDynamicsConfig
from uav_rendezvous_rl.observations import ObservationPipelineCfg, make_m7a_observation_cfg

from .uav_rendezvous_recurrent_env_cfg import UavRendezvousRecurrentEnvCfg


@configclass
class UavRendezvousM7BEnvCfg(UavRendezvousRecurrentEnvCfg):
    """M7B task config with dynamics randomization, action delay, and wind."""

    state_space = 65
    observation_degradation: ObservationPipelineCfg = make_m7a_observation_cfg(0)
    m7b_dynamics: M7BDynamicsConfig = M7BDynamicsConfig()

    tau_velocity_scale: float = 1.0
    acceleration_limit_scale: float = 1.0
    speed_limit_scale: float = 1.0
    linear_drag: float = 0.0
    action_delay_steps: int = 0
    steady_wind_max_magnitude: float = 0.0
    gust_max_magnitude: float = 0.0
