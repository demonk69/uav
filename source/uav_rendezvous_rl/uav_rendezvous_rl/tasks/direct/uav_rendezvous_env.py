"""M1 minimal DirectRLEnv task.

The environment is a lifecycle smoke target only. It does not model dual-UAV
truth state, target motion, offset rewards, baselines, PPO, GRU, or privileged
critic observations.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .uav_rendezvous_env_cfg import UavRendezvousEnvCfg


class UavRendezvousEnv(DirectRLEnv):
    """Minimal vectorized DirectRLEnv for M1 smoke tests."""

    cfg: UavRendezvousEnvCfg

    def __init__(self, cfg: UavRendezvousEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros((self.num_envs, self.single_action_space.shape[0]), device=self.device)
        self._placeholder_state_w = torch.zeros((self.num_envs, 2), device=self.device)

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # Placeholder single-axis state update to exercise reset/step without M2 UAV logic.
        self._placeholder_state_w[:, 1] = self._actions[:, 0]
        self._placeholder_state_w[:, 0] = torch.clamp(
            self._placeholder_state_w[:, 0] + self._placeholder_state_w[:, 1] * self.physics_dt,
            -10.0,
            10.0,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        progress = self.episode_length_buf.to(dtype=torch.float32).unsqueeze(-1) / float(self.max_episode_length)
        obs = torch.cat((self._placeholder_state_w, progress, self._actions), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        non_finite = torch.any(~torch.isfinite(self._placeholder_state_w), dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return non_finite, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._placeholder_state_w[env_ids] = 0.0
