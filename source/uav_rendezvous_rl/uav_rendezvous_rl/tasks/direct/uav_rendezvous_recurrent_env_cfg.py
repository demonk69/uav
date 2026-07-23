"""Configuration for the independent M6 recurrent rendezvous task."""

from __future__ import annotations

from dataclasses import replace

from isaaclab.utils import configclass
from uav_rendezvous_rl.motions import TargetMotionManagerCfg

from .uav_rendezvous_rl_env_cfg import UavRendezvousRLEnvCfg


def _mixed_motion_cfg() -> TargetMotionManagerCfg:
    """Return the controlled four-mode M6 target-motion distribution."""

    cfg = TargetMotionManagerCfg()
    mixed = (0.25, 0.25, 0.25, 0.25)
    return replace(
        cfg,
        force_mode_cycle_on_reset=False,
        train=replace(cfg.train, mode_probabilities=mixed),
        validation=replace(cfg.validation, mode_probabilities=mixed),
        test=replace(cfg.test, mode_probabilities=mixed),
    )


@configclass
class UavRendezvousRecurrentEnvCfg(UavRendezvousRLEnvCfg):
    """M6 recurrent task config, separate from the M5 feedforward task."""

    # M6 keeps the fixed M5 3D raw velocity action and 25D/57D observation contract.
    target_motion: TargetMotionManagerCfg = _mixed_motion_cfg()
    target_motion_split = "train"
