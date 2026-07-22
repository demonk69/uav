"""Configuration for the independent M4 deterministic baseline task."""

from __future__ import annotations

from isaaclab.utils import configclass
from uav_rendezvous_rl.controllers import BaselineControllerCfg, BaselineInitialGeometryCfg

from .uav_rendezvous_env_cfg import UavRendezvousEnvCfg


@configclass
class UavRendezvousBaselineEnvCfg(UavRendezvousEnvCfg):
    """M4 baseline task config.

    This task is intentionally separate from `Isaac-Uav-Rendezvous-Direct-v0` so
    M2/M3 regression behavior remains unchanged.
    """

    episode_length_s = 20.0

    baseline_controller: BaselineControllerCfg = BaselineControllerCfg()
    baseline_initial_geometry: BaselineInitialGeometryCfg = BaselineInitialGeometryCfg()

    workspace_x_range = (-250.0, 250.0)
    workspace_y_range = (-250.0, 250.0)
    workspace_z_range = (1.5, 1.5)
