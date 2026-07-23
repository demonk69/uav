"""Independent M6 recurrent RL DirectRLEnv task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uav_rendezvous_rl.motions.configs import MODE_NAMES

from .uav_rendezvous_recurrent_env_cfg import UavRendezvousRecurrentEnvCfg
from .uav_rendezvous_rl_env import UavRendezvousRLEnv


class UavRendezvousRecurrentEnv(UavRendezvousRLEnv):
    """M6 task reusing M5 dynamics with mixed target modes for recurrent PPO."""

    cfg: UavRendezvousRecurrentEnvCfg

    def __init__(self, cfg: UavRendezvousRecurrentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _record_episode_metrics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        start_index = len(self._episode_history)
        super()._record_episode_metrics(env_ids_tensor)
        finished_env_ids = env_ids_tensor.detach().cpu().tolist()
        for entry, env_id in zip(self._episode_history[start_index:], finished_env_ids, strict=True):
            mode_id = int(self.target_motion_manager.mode_id[env_id].item())
            entry["target_motion_mode_id"] = mode_id
            entry["target_motion_mode"] = MODE_NAMES[mode_id]

    def get_m6_episode_history(self, clear: bool = False) -> list[dict[str, float | int | bool | str]]:
        return self.get_m5_episode_history(clear=clear)

    def get_m6_diagnostics(self) -> dict[str, object]:
        diagnostics = self.get_m5_diagnostics()
        diagnostics["mode_counts"] = self.target_motion_manager.mode_counts()
        diagnostics["target_motion_split"] = self.cfg.target_motion_split
        return diagnostics
