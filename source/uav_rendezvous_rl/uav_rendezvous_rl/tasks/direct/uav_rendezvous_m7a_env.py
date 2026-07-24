"""Independent M7A partial-observability DirectRLEnv task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from uav_rendezvous_rl.mdp import assemble_actor_observation, assemble_critic_observation
from uav_rendezvous_rl.observations import ObservationPipeline

from .m2_kinematics import all_finite
from .uav_rendezvous_m7a_env_cfg import UavRendezvousM7AEnvCfg
from .uav_rendezvous_recurrent_env import UavRendezvousRecurrentEnv


class UavRendezvousM7AEnv(UavRendezvousRecurrentEnv):
    """M7A task using M6 dynamics with strictly causal degraded relative observations."""

    cfg: UavRendezvousM7AEnvCfg

    def __init__(self, cfg: UavRendezvousM7AEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.observation_pipeline = ObservationPipeline(
            self.cfg.observation_degradation,
            self.num_envs,
            self.device,
            seed=int(self.cfg.seed if self.cfg.seed is not None else 0),
        )
        self.p_rel_obs_w = torch.zeros_like(self.p_rel_w)
        self.v_rel_obs_w = torch.zeros_like(self.v_rel_w)
        self.observation_pipeline.reset(self._all_env_ids, self.p_rel_w, self.v_rel_w)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        resolved_env_ids = self._resolve_env_ids(env_ids)
        super()._reset_idx(resolved_env_ids)
        if hasattr(self, "observation_pipeline"):
            self.observation_pipeline.reset(resolved_env_ids, self.p_rel_w[resolved_env_ids], self.v_rel_w[resolved_env_ids])
            self.p_rel_obs_w[resolved_env_ids] = self.p_rel_w[resolved_env_ids]
            self.v_rel_obs_w[resolved_env_ids] = self.v_rel_w[resolved_env_ids]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._write_entities_to_sim(self._all_env_ids)
        self.p_rel_obs_w[:], self.v_rel_obs_w[:] = self.observation_pipeline.observe(self.p_rel_w, self.v_rel_w)
        actor_obs = assemble_actor_observation(
            self.p_rel_obs_w,
            self.v_rel_obs_w,
            self.v_ego_w,
            self._r_ego_6d,
            self._omega_ego_b,
            self.previous_squashed_action,
            self.b_des_w,
            self.cfg.d_offset,
        )
        mode_one_hot = F.one_hot(self.target_motion_manager.mode_id, num_classes=4).to(dtype=torch.float32)
        target_motion_current_params = self._target_motion_current_params()
        episode_phase = torch.clamp(self.episode_length_buf.to(dtype=torch.float32) / float(self.max_episode_length), 0.0, 1.0)
        critic_obs = assemble_critic_observation(
            actor_obs,
            self.p_ego_w,
            self.p_target_w,
            self.v_target_w,
            self.a_target_w,
            self._r_target_6d,
            self._omega_target_b,
            mode_one_hot,
            target_motion_current_params,
            episode_phase,
        )
        return {"policy": actor_obs, "critic": critic_obs}

    def get_m7a_diagnostics(self) -> dict[str, object]:
        policy_obs = self.obs_buf.get("policy", None) if isinstance(getattr(self, "obs_buf", None), dict) else None
        critic_obs = self.obs_buf.get("critic", None) if isinstance(getattr(self, "obs_buf", None), dict) else None
        if policy_obs is None or critic_obs is None:
            actor_obs = assemble_actor_observation(
                self.p_rel_obs_w,
                self.v_rel_obs_w,
                self.v_ego_w,
                self._r_ego_6d,
                self._omega_ego_b,
                self.previous_squashed_action,
                self.b_des_w,
                self.cfg.d_offset,
            )
            mode_one_hot = F.one_hot(self.target_motion_manager.mode_id, num_classes=4).to(dtype=torch.float32)
            episode_phase = torch.clamp(
                self.episode_length_buf.to(dtype=torch.float32) / float(self.max_episode_length), 0.0, 1.0
            )
            critic_obs = assemble_critic_observation(
                actor_obs,
                self.p_ego_w,
                self.p_target_w,
                self.v_target_w,
                self.a_target_w,
                self._r_target_6d,
                self._omega_target_b,
                mode_one_hot,
                self._target_motion_current_params(),
                episode_phase,
            )
            policy_obs = actor_obs
        finite = all_finite(
            self.p_ego_w,
            self.v_ego_w,
            self.p_target_w,
            self.v_target_w,
            self.a_target_w,
            self.p_rel_w,
            self.v_rel_w,
            self.e_offset_w,
            self.p_rel_obs_w,
            self.v_rel_obs_w,
            policy_obs,
            critic_obs,
        )
        return {
            "num_envs": int(self.num_envs),
            "policy_obs_dim": int(policy_obs.shape[1]),
            "critic_obs_dim": int(critic_obs.shape[1]),
            "finite_check": finite,
            "mode_counts": self.target_motion_manager.mode_counts(),
            "target_motion_split": self.cfg.target_motion_split,
            "observation_pipeline": self.observation_pipeline.diagnostics(),
        }

    def get_m7a_episode_history(self, clear: bool = False) -> list[dict[str, float | int | bool | str]]:
        return self.get_m6_episode_history(clear=clear)
