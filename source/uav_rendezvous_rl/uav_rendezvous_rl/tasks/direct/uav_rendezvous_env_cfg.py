"""Configuration for the M2 dual-placeholder DirectRLEnv task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class UavRendezvousEnvCfg(DirectRLEnvCfg):
    """M2 environment config.

    The 6D observation `[p_rel_w, v_rel_w]` is only the M2 acceptance
    interface. It is not the final M5 Actor observation definition.
    """

    # env
    seed = 42
    decimation = 2
    episode_length_s = 10.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = 6
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

    # M2 placeholder entity geometry and safety values [m]
    r_ego = 0.20
    r_target = 0.20
    safety_margin = 0.35
    d_safe = 0.75

    # M2 deterministic ego state in env-local world frame w [m, m/s]
    ego_initial_pos_w = (0.0, 0.0, 1.5)

    # M2 target reset randomization ranges in env-local world frame w [m, m/s]
    target_pos_x_range = (4.0, 8.0)
    target_pos_y_range = (-2.0, 2.0)
    target_height_range = (1.5, 1.5)
    target_vel_x_range = (0.2, 1.0)
    target_vel_y_range = (-0.5, 0.5)

    # M1/M2 fixed desired offset definition retained for e_offset_w [m]
    d_offset = 5.0
    b_des_w = (5.0, 0.0, 0.0)
    ego_static_tolerance = 1.0e-6

    # Placeholder assets are synchronized carriers for task tensors, not the source of truth.
    ego_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ego",
        spawn=sim_utils.SphereCfg(
            radius=r_ego,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=ego_initial_pos_w, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=r_target,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )
