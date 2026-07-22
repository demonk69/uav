"""Configuration for the independent M5 feedforward RL rendezvous task."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np

from isaaclab.utils import configclass
from uav_rendezvous_rl.mdp import RendezvousActionCfg, RendezvousInitialGeometryCfg, RendezvousRewardCfg
from uav_rendezvous_rl.motions import TargetMotionManagerCfg

from .uav_rendezvous_env_cfg import UavRendezvousEnvCfg


def _constant_velocity_motion_cfg() -> TargetMotionManagerCfg:
    """Return a CV-only target-motion config for formal M5 PPO training."""

    cfg = TargetMotionManagerCfg()
    cv_only = (1.0, 0.0, 0.0, 0.0)
    return replace(
        cfg,
        force_mode_cycle_on_reset=False,
        train=replace(cfg.train, mode_probabilities=cv_only),
        validation=replace(cfg.validation, mode_probabilities=cv_only),
        test=replace(cfg.test, mode_probabilities=cv_only),
    )


@configclass
class UavRendezvousRLEnvCfg(UavRendezvousEnvCfg):
    """M5 RL task config, separate from Direct and Baseline regression tasks."""

    # env
    episode_length_s = 20.0
    action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    observation_space = 25
    state_space = 57

    # Formal M5 PPO uses ConstantVelocity only. Runtime audits may override this to mixed modes.
    target_motion: TargetMotionManagerCfg = _constant_velocity_motion_cfg()
    target_motion_split = "train"

    # M5 randomized desired offset and simplified ego dynamics.
    action: RendezvousActionCfg = RendezvousActionCfg()
    initial_geometry: RendezvousInitialGeometryCfg = RendezvousInitialGeometryCfg()
    reward: RendezvousRewardCfg = RendezvousRewardCfg()

    # Target resets at 1.5 m, but the 3D velocity-action ego needs a non-degenerate safe altitude band.
    workspace_x_range = (-250.0, 250.0)
    workspace_y_range = (-250.0, 250.0)
    workspace_z_range = (0.5, 5.0)
