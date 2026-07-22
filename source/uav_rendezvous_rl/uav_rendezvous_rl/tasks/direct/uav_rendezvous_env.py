"""M2 dual-placeholder DirectRLEnv task.

The environment task tensors are the single source of truth for M2. The ego and
target RigidObjects are synchronized carriers for state inspection and
visualization. The task intentionally implements only the M2 scope: a stationary
ego placeholder and a fixed-height, constant-velocity target placeholder. It
does not implement M3 target motion generators, baselines, PPO, recurrent
policies, asymmetric actor-critic, or high-fidelity multirotor dynamics.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .m2_kinematics import (
    M2RandomizationCfg,
    all_finite,
    compute_constant_velocity_position_w,
    compute_offset_error_w,
    compute_relative_state_w,
    sample_m2_initial_conditions,
)
from .uav_rendezvous_env_cfg import UavRendezvousEnvCfg


class UavRendezvousEnv(DirectRLEnv):
    """Vectorized DirectRLEnv for M2 truth-state smoke tests."""

    cfg: UavRendezvousEnvCfg

    def __init__(self, cfg: UavRendezvousEnvCfg, render_mode: str | None = None, **kwargs):
        self._rng = torch.Generator(device="cpu")
        self._seed_value = int(cfg.seed if cfg.seed is not None else 0)
        self._rng.manual_seed(self._seed_value)
        super().__init__(cfg, render_mode, **kwargs)

        self._all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._actions = torch.zeros((self.num_envs, self.single_action_space.shape[0]), device=self.device)
        self._identity_quat_w = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._zero_ang_vel_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._m2_randomization_cfg = M2RandomizationCfg(
            ego_initial_pos_w=self.cfg.ego_initial_pos_w,
            target_pos_x_range=self.cfg.target_pos_x_range,
            target_pos_y_range=self.cfg.target_pos_y_range,
            target_height_range=self.cfg.target_height_range,
            target_vel_x_range=self.cfg.target_vel_x_range,
            target_vel_y_range=self.cfg.target_vel_y_range,
            d_safe=self.cfg.d_safe,
        )

        self.p_ego_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.v_ego_w = torch.zeros_like(self.p_ego_w)
        self.p_target_initial_w = torch.zeros_like(self.p_ego_w)
        self.p_target_w = torch.zeros_like(self.p_ego_w)
        self.v_target_w = torch.zeros_like(self.p_ego_w)
        self.p_rel_w = torch.zeros_like(self.p_ego_w)
        self.v_rel_w = torch.zeros_like(self.p_ego_w)
        self.b_des_w = torch.tensor(self.cfg.b_des_w, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self._ego_initial_pos_w = torch.tensor(self.cfg.ego_initial_pos_w, dtype=torch.float32, device=self.device)
        self.e_offset_w = torch.zeros_like(self.p_ego_w)
        self.target_elapsed_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_displacement_max_observed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ego_displacement_max_observed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _setup_scene(self) -> None:
        self.ego = RigidObject(self.cfg.ego_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["ego"] = self.ego
        self.scene.rigid_objects["target"] = self.target
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def seed(self, seed: int = -1) -> int:
        resolved_seed = super().seed(seed)
        if hasattr(self, "_rng"):
            self._rng.manual_seed(resolved_seed)
            self._seed_value = int(resolved_seed)
        return resolved_seed

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # Actions are intentionally a no-op in M2. Target motion is analytic to avoid integration drift.
        self.target_elapsed_time += self.physics_dt
        self.p_target_w[:] = compute_constant_velocity_position_w(
            self.p_target_initial_w, self.v_target_w, self.target_elapsed_time
        )
        self._refresh_relative_state()
        self._update_motion_diagnostics()
        self._write_entities_to_sim(self._all_env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # M2 acceptance observation only: this is not the final M5 Actor observation definition.
        self._write_entities_to_sim(self._all_env_ids)
        obs = torch.cat((self.p_rel_w, self.v_rel_w), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        state_tensors = torch.cat(
            (self.p_ego_w, self.v_ego_w, self.p_target_w, self.v_target_w, self.p_rel_w, self.v_rel_w), dim=-1
        )
        non_finite = torch.any(~torch.isfinite(state_tensors), dim=1)
        collision_risk = torch.linalg.norm(self.p_rel_w, dim=1) < self.cfg.d_safe
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return non_finite | collision_risk, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        super()._reset_idx(env_ids)

        num_reset = int(env_ids.numel())
        p_ego_w, v_ego_w, p_target_initial_w, v_target_w = sample_m2_initial_conditions(
            num_reset,
            self._m2_randomization_cfg,
            self._rng,
            self.device,
        )

        self._actions[env_ids] = 0.0
        self.p_ego_w[env_ids] = p_ego_w
        self.v_ego_w[env_ids] = v_ego_w
        self.p_target_initial_w[env_ids] = p_target_initial_w
        self.p_target_w[env_ids] = p_target_initial_w
        self.v_target_w[env_ids] = v_target_w
        self.target_elapsed_time[env_ids] = 0.0
        self.reset_counts[env_ids] += 1

        self._refresh_relative_state()
        self._write_entities_to_sim(env_ids)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return self._all_env_ids
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def _refresh_relative_state(self) -> None:
        self.p_rel_w[:], self.v_rel_w[:] = compute_relative_state_w(
            self.p_ego_w, self.v_ego_w, self.p_target_w, self.v_target_w
        )
        self.e_offset_w[:] = compute_offset_error_w(self.p_ego_w, self.p_target_w, self.b_des_w)

    def _local_to_sim_world(self, p_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        return p_w[env_ids] + self.scene.env_origins[env_ids]

    def _write_entities_to_sim(self, env_ids: torch.Tensor) -> None:
        ego_pose = torch.cat((self._local_to_sim_world(self.p_ego_w, env_ids), self._identity_quat_w[env_ids]), dim=-1)
        target_pose = torch.cat(
            (self._local_to_sim_world(self.p_target_w, env_ids), self._identity_quat_w[env_ids]), dim=-1
        )
        ego_vel = torch.cat((self.v_ego_w[env_ids], self._zero_ang_vel_w[env_ids]), dim=-1)
        target_vel = torch.cat((self.v_target_w[env_ids], self._zero_ang_vel_w[env_ids]), dim=-1)

        self.ego.write_root_pose_to_sim(ego_pose, env_ids=env_ids)
        self.ego.write_root_velocity_to_sim(ego_vel, env_ids=env_ids)
        self.target.write_root_pose_to_sim(target_pose, env_ids=env_ids)
        self.target.write_root_velocity_to_sim(target_vel, env_ids=env_ids)

    def _update_motion_diagnostics(self) -> None:
        target_displacement = torch.linalg.norm(self.p_target_w - self.p_target_initial_w, dim=1)
        ego_displacement = torch.linalg.norm(self.p_ego_w - self._ego_initial_pos_w.unsqueeze(0), dim=1)
        self._target_displacement_max_observed[:] = torch.maximum(
            self._target_displacement_max_observed, target_displacement
        )
        self._ego_displacement_max_observed[:] = torch.maximum(self._ego_displacement_max_observed, ego_displacement)

    def get_m2_diagnostics(self) -> dict[str, object]:
        """Return M2 smoke-test diagnostics with CPU scalar/list values."""

        relative_distance = torch.linalg.norm(self.p_rel_w, dim=1)
        target_speed = torch.linalg.norm(self.v_target_w, dim=1)
        finite_check = all_finite(
            self.p_ego_w,
            self.v_ego_w,
            self.p_target_w,
            self.v_target_w,
            self.p_rel_w,
            self.v_rel_w,
            self.e_offset_w,
        )
        max_target_motion = float(self._target_displacement_max_observed.max().item())
        max_ego_motion = float(self._ego_displacement_max_observed.max().item())
        return {
            "num_envs": int(self.num_envs),
            "total_steps": int(self.common_step_counter),
            "target_position_min": self.p_target_w.min(dim=0).values.detach().cpu().tolist(),
            "target_position_max": self.p_target_w.max(dim=0).values.detach().cpu().tolist(),
            "target_speed_min": float(target_speed.min().item()),
            "target_speed_max": float(target_speed.max().item()),
            "relative_distance_min": float(relative_distance.min().item()),
            "relative_distance_max": float(relative_distance.max().item()),
            "finite_check": finite_check,
            "reset_counts": self.reset_counts.detach().cpu().tolist(),
            "seed": int(self._seed_value),
            "target_displacement_max_observed": max_target_motion,
            "ego_displacement_max_observed": max_ego_motion,
            "target_moved": max_target_motion > 1.0e-6,
            "ego_static": max_ego_motion <= self.cfg.ego_static_tolerance,
        }
