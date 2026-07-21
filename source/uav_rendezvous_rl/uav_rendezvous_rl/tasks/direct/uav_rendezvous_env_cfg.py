"""Configuration for the M1 minimal DirectRLEnv task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class UavRendezvousEnvCfg(DirectRLEnvCfg):
    """Minimal M1 environment config.

    This config is intentionally not the M2 dual-UAV truth environment.
    """

    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = 4
    state_space = 0
    wait_for_textures = False
    ui_window_class_type = None

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
