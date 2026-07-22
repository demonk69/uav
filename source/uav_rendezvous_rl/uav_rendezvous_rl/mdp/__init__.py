"""Pure PyTorch MDP helpers for UAV rendezvous tasks."""

from .rendezvous import (
    RendezvousActionCfg,
    RendezvousInitialGeometry,
    RendezvousInitialGeometryCfg,
    RendezvousRewardCfg,
    assemble_actor_observation,
    assemble_critic_observation,
    compute_reward_terms,
    encode_target_motion_current_params,
    identity_rotation_6d,
    map_raw_action_to_velocity_command,
    raw_action_from_velocity_command,
    sample_m5_initial_geometry,
    validate_m5_initial_geometry,
)

__all__ = [
    "RendezvousActionCfg",
    "RendezvousInitialGeometry",
    "RendezvousInitialGeometryCfg",
    "RendezvousRewardCfg",
    "assemble_actor_observation",
    "assemble_critic_observation",
    "compute_reward_terms",
    "encode_target_motion_current_params",
    "identity_rotation_6d",
    "map_raw_action_to_velocity_command",
    "raw_action_from_velocity_command",
    "sample_m5_initial_geometry",
    "validate_m5_initial_geometry",
]
