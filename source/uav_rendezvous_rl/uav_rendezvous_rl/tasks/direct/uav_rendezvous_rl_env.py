"""Independent M5 feedforward RL DirectRLEnv task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv

from uav_rendezvous_rl.controllers import compute_limited_acceleration, integrate_ego_kinematics
from uav_rendezvous_rl.mdp import (
    assemble_actor_observation,
    assemble_critic_observation,
    compute_reward_terms,
    encode_target_motion_current_params,
    identity_rotation_6d,
    map_raw_action_to_velocity_command,
    sample_m5_initial_geometry,
)
from uav_rendezvous_rl.motions.configs import get_split_cfg
from uav_rendezvous_rl.motions.sampling import uniform_range

from .m2_kinematics import all_finite
from .m4_accounting import update_collision_risk_accounting
from .uav_rendezvous_env import UavRendezvousEnv
from .uav_rendezvous_rl_env_cfg import UavRendezvousRLEnvCfg


class UavRendezvousRLEnv(UavRendezvousEnv):
    """M5 velocity-action RL environment with asymmetric feedforward observations."""

    cfg: UavRendezvousRLEnvCfg

    def __init__(self, cfg: UavRendezvousRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._action_cfg = self.cfg.action
        self._initial_geometry_cfg = self.cfg.initial_geometry
        self._reward_cfg = self.cfg.reward
        self._success_hold_steps_required = max(1, int(round(self._reward_cfg.success_hold_s / self.step_dt)))

        action_dim = int(self.single_action_space.shape[0])
        self.raw_action = torch.zeros((self.num_envs, action_dim), dtype=torch.float32, device=self.device)
        self.squashed_action = torch.zeros_like(self.raw_action)
        self.previous_squashed_action = torch.zeros_like(self.raw_action)
        self._action_delta_squashed = torch.zeros_like(self.raw_action)
        self.v_cmd_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.a_cmd_w = torch.zeros_like(self.v_cmd_w)

        self._r_ego_6d = identity_rotation_6d(self.num_envs, self.device)
        self._r_target_6d = identity_rotation_6d(self.num_envs, self.device)
        self._omega_ego_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._omega_target_b = torch.zeros_like(self._omega_ego_b)

        self.theta_offset = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.delta_radial = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.delta_tangent = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._previous_offset_error_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.workspace_violation_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.height_violation_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.speed_violation_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.attitude_violation_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nan_or_inf_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_hold_completed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_step_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_completed_step_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_hold_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.collision_risk_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.workspace_violation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.height_violation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_saturation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.speed_limit_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.acceleration_saturation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rl_policy_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rl_physics_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._acceleration_saturated_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._speed_saturated_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.initial_offset_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.final_offset_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.minimum_offset_error = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self.final_relative_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.minimum_center_distance = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self.convergence_time = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)
        self.success_offset_error = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)
        self.success_relative_speed = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)
        self._ego_speed_max_observed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ego_acceleration_max_observed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._latest_reward_terms: dict[str, torch.Tensor] = {}
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in (
                "offset",
                "relative_velocity",
                "progress",
                "action_smoothness",
                "action_magnitude",
                "safety_distance",
                "speed_limit",
                "accel_limit",
                "attitude_rate",
                "workspace",
                "success_bonus",
            )
        }
        self._episode_history: list[dict[str, float | int | bool]] = []

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 3:
            raise RuntimeError(f"M5 RL action must have shape (num_envs, 3), got {tuple(actions.shape)}.")
        non_finite = ~torch.isfinite(actions).all(dim=1)
        self.nan_or_inf_buf[:] |= non_finite
        self.raw_action[:] = torch.nan_to_num(actions.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        squashed_action, v_cmd_w = map_raw_action_to_velocity_command(self.raw_action, self._action_cfg.v_max)
        self._action_delta_squashed[:] = squashed_action - self.previous_squashed_action
        self.squashed_action[:] = squashed_action
        self.v_cmd_w[:] = v_cmd_w
        self.previous_squashed_action[:] = squashed_action
        self.action_saturation_count += torch.any(torch.abs(squashed_action) >= 0.95, dim=1).to(torch.long)
        self.rl_policy_step_count += 1
        self._acceleration_saturated_step[:] = False
        self._speed_saturated_step[:] = False

    def _apply_action(self) -> None:
        # Causality: ego control uses target state at the start of this physics substep.
        self.a_cmd_w[:], acceleration_saturated = compute_limited_acceleration(
            self.v_cmd_w, self.v_ego_w, self._action_cfg.tau_v, self._action_cfg.a_max
        )
        p_ego_next_w, v_ego_next_w, speed_saturated = integrate_ego_kinematics(
            self.p_ego_w, self.v_ego_w, self.a_cmd_w, self.physics_dt, self._action_cfg.v_abs_max
        )
        self.p_ego_w[:] = p_ego_next_w
        self.v_ego_w[:] = v_ego_next_w
        self._check_collision_against_current_target()

        target_state = self.target_motion_manager.step()
        self.p_target_w[:] = target_state.p_target_w
        self.v_target_w[:] = target_state.v_target_w
        self.a_target_w[:] = target_state.a_target_w
        self.target_elapsed_time[:] = self.motion_step_count.to(dtype=torch.float32) * self.physics_dt

        self._acceleration_saturated_step[:] |= acceleration_saturated
        self._speed_saturated_step[:] |= speed_saturated
        self.acceleration_saturation_count += acceleration_saturated.to(torch.long)
        self.speed_limit_count += speed_saturated.to(torch.long)
        self.rl_physics_step_count += 1
        self._ego_speed_max_observed[:] = torch.maximum(self._ego_speed_max_observed, torch.linalg.norm(self.v_ego_w, dim=1))
        self._ego_acceleration_max_observed[:] = torch.maximum(
            self._ego_acceleration_max_observed, torch.linalg.norm(self.a_cmd_w, dim=1)
        )

        self._refresh_relative_state()
        self._update_rl_diagnostics()
        self._update_motion_diagnostics()
        self._write_entities_to_sim(self._all_env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._write_entities_to_sim(self._all_env_ids)
        actor_obs = assemble_actor_observation(
            self.p_rel_w,
            self.v_rel_w,
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

    def _get_rewards(self) -> torch.Tensor:
        center_distance = torch.linalg.norm(self.p_rel_w, dim=1)
        reward, terms, offset_error_norm = compute_reward_terms(
            self.e_offset_w,
            self._previous_offset_error_norm,
            self.v_ego_w,
            self.v_target_w,
            self.raw_action,
            self._action_delta_squashed,
            center_distance,
            self._acceleration_saturated_step,
            self._speed_saturated_step,
            self.collision_risk_buf,
            self.workspace_violation_buf | self.height_violation_buf,
            self._success_step_buf,
            self._success_completed_step_buf,
            self.cfg.d_safe,
            self._omega_ego_b,
            self._reward_cfg,
        )
        self._previous_offset_error_norm[:] = offset_error_norm
        self._latest_reward_terms = terms
        for key, value in terms.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_success_diagnostics()
        state_tensors = torch.cat(
            (
                self.p_ego_w,
                self.v_ego_w,
                self.p_target_w,
                self.v_target_w,
                self.a_target_w,
                self.p_rel_w,
                self.v_rel_w,
                self.e_offset_w,
                self.raw_action,
                self.squashed_action,
                self.v_cmd_w,
                self.a_cmd_w,
                self.b_des_w,
            ),
            dim=-1,
        )
        self.nan_or_inf_buf[:] |= torch.any(~torch.isfinite(state_tensors), dim=1)
        terminated = (
            self.collision_risk_buf
            | self.workspace_violation_buf
            | self.height_violation_buf
            | self.speed_violation_buf
            | self.attitude_violation_buf
            | self.nan_or_inf_buf
            | self.target_motion_invalid_buf
        )
        if self._reward_cfg.terminate_on_success:
            terminated |= self.success_hold_completed_buf
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if hasattr(self, "_episode_sums"):
            self._record_episode_log(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)

        num_reset = int(env_ids.numel())
        split_cfg = get_split_cfg(self.cfg.target_motion, self.cfg.target_motion_split)
        p_target_initial_w = torch.zeros((num_reset, 3), dtype=torch.float32, device=self.device)
        v_target_initial_w = torch.zeros_like(p_target_initial_w)
        p_target_initial_w[:, 0] = uniform_range(split_cfg.target_pos_x_range, (num_reset,), self._rng, self.device)
        p_target_initial_w[:, 1] = uniform_range(split_cfg.target_pos_y_range, (num_reset,), self._rng, self.device)
        p_target_initial_w[:, 2] = uniform_range(split_cfg.target_height_range, (num_reset,), self._rng, self.device)
        v_target_initial_w[:, 0] = uniform_range(split_cfg.target_vel_x_range, (num_reset,), self._rng, self.device)
        v_target_initial_w[:, 1] = uniform_range(split_cfg.target_vel_y_range, (num_reset,), self._rng, self.device)

        geometry = sample_m5_initial_geometry(
            p_target_initial_w,
            self.cfg.d_offset,
            self.cfg.d_safe,
            self._initial_geometry_cfg,
            self._rng,
            self.device,
        )
        target_state = self.target_motion_manager.reset(
            env_ids, p_target_initial_w, v_target_initial_w, self._rng, self.cfg.target_motion_split
        )

        self._actions[env_ids] = 0.0
        self.raw_action[env_ids] = 0.0
        self.squashed_action[env_ids] = 0.0
        self.previous_squashed_action[env_ids] = 0.0
        self._action_delta_squashed[env_ids] = 0.0
        self.v_cmd_w[env_ids] = 0.0
        self.a_cmd_w[env_ids] = 0.0
        self.p_ego_w[env_ids] = geometry.p_ego_w
        self.v_ego_w[env_ids] = geometry.v_ego_w
        self.b_des_w[env_ids] = geometry.b_des_w
        self.theta_offset[env_ids] = geometry.theta
        self.delta_radial[env_ids] = geometry.delta_radial
        self.delta_tangent[env_ids] = geometry.delta_tangent
        self.p_target_initial_w[env_ids] = p_target_initial_w
        self.p_target_w[env_ids] = target_state.p_target_w[env_ids]
        self.v_target_w[env_ids] = target_state.v_target_w[env_ids]
        self.a_target_w[env_ids] = target_state.a_target_w[env_ids]
        self.target_elapsed_time[env_ids] = 0.0
        self.target_motion_invalid_buf[env_ids] = self.target_motion_manager.invalid_mask[env_ids]

        self._refresh_relative_state()
        offset_error = torch.linalg.norm(self.e_offset_w[env_ids], dim=1)
        center_distance = torch.linalg.norm(self.p_rel_w[env_ids], dim=1)
        relative_speed = torch.linalg.norm(self.v_ego_w[env_ids] - self.v_target_w[env_ids], dim=1)

        self.workspace_violation_buf[env_ids] = False
        self.height_violation_buf[env_ids] = False
        self.speed_violation_buf[env_ids] = False
        self.attitude_violation_buf[env_ids] = False
        self.nan_or_inf_buf[env_ids] = False
        self.collision_risk_buf[env_ids] = False
        self.success_buf[env_ids] = False
        self.success_hold_completed_buf[env_ids] = False
        self._success_step_buf[env_ids] = False
        self._success_completed_step_buf[env_ids] = False
        self.success_hold_step_count[env_ids] = 0
        self.collision_risk_count[env_ids] = 0
        self.workspace_violation_count[env_ids] = 0
        self.height_violation_count[env_ids] = 0
        self.action_saturation_count[env_ids] = 0
        self.speed_limit_count[env_ids] = 0
        self.acceleration_saturation_count[env_ids] = 0
        self.rl_policy_step_count[env_ids] = 0
        self.rl_physics_step_count[env_ids] = 0
        self._acceleration_saturated_step[env_ids] = False
        self._speed_saturated_step[env_ids] = False
        self.initial_offset_error[env_ids] = offset_error
        self.final_offset_error[env_ids] = offset_error
        self.minimum_offset_error[env_ids] = offset_error
        self.final_relative_speed[env_ids] = relative_speed
        self.minimum_center_distance[env_ids] = center_distance
        self.convergence_time[env_ids] = float("nan")
        self.success_offset_error[env_ids] = float("nan")
        self.success_relative_speed[env_ids] = float("nan")
        self.min_relative_distance_per_episode[env_ids] = center_distance
        self._previous_offset_error_norm[env_ids] = offset_error
        self._ego_speed_max_observed[env_ids] = torch.linalg.norm(self.v_ego_w[env_ids], dim=1)
        self._ego_acceleration_max_observed[env_ids] = 0.0
        self._workspace_abs_max_observed[env_ids] = torch.maximum(
            torch.abs(self.p_ego_w[env_ids]).amax(dim=1), torch.abs(self.p_target_w[env_ids]).amax(dim=1)
        )
        self._target_speed_max_observed[env_ids] = torch.linalg.norm(self.v_target_w[env_ids], dim=1)
        self._target_acceleration_max_observed[env_ids] = torch.linalg.norm(self.a_target_w[env_ids], dim=1)
        for episode_sum in self._episode_sums.values():
            episode_sum[env_ids] = 0.0
        self.reset_counts[env_ids] += 1
        self._update_rl_diagnostics(env_ids)
        self._write_entities_to_sim(env_ids)

    def _check_collision_against_current_target(self) -> None:
        center_distance = torch.linalg.norm(self.p_target_w - self.p_ego_w, dim=1)
        collision = center_distance < self.cfg.d_safe
        update_collision_risk_accounting(collision, self.collision_risk_buf, self.collision_risk_count)
        self.minimum_center_distance[:] = torch.minimum(self.minimum_center_distance, center_distance)
        self.min_relative_distance_per_episode[:] = torch.minimum(self.min_relative_distance_per_episode, center_distance)
        self._min_relative_distance_observed[:] = torch.minimum(self._min_relative_distance_observed, center_distance)

    def _update_rl_diagnostics(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self._all_env_ids
        center_distance = torch.linalg.norm(self.p_rel_w[env_ids], dim=1)
        offset_error = torch.linalg.norm(self.e_offset_w[env_ids], dim=1)
        relative_speed = torch.linalg.norm(self.v_ego_w[env_ids] - self.v_target_w[env_ids], dim=1)
        speed = torch.linalg.norm(self.v_ego_w[env_ids], dim=1)
        workspace_valid = self._workspace_valid(env_ids)
        height_valid = self._height_valid(env_ids)
        speed_valid = speed <= self._action_cfg.v_abs_max + 1.0e-5
        collision = center_distance < self.cfg.d_safe

        self.final_offset_error[env_ids] = offset_error
        self.minimum_offset_error[env_ids] = torch.minimum(self.minimum_offset_error[env_ids], offset_error)
        self.final_relative_speed[env_ids] = relative_speed
        self.minimum_center_distance[env_ids] = torch.minimum(self.minimum_center_distance[env_ids], center_distance)
        self.min_relative_distance_per_episode[env_ids] = torch.minimum(
            self.min_relative_distance_per_episode[env_ids], center_distance
        )
        self._min_relative_distance_observed[env_ids] = torch.minimum(
            self._min_relative_distance_observed[env_ids], center_distance
        )
        previous_collision = self.collision_risk_buf[env_ids]
        new_collision = collision & ~previous_collision
        self.collision_risk_buf[env_ids] = previous_collision | collision
        self.collision_risk_count[env_ids] += new_collision.to(torch.long)
        workspace_violation = ~workspace_valid
        height_violation = ~height_valid
        speed_violation = ~speed_valid
        self.workspace_violation_buf[env_ids] |= workspace_violation
        self.height_violation_buf[env_ids] |= height_violation
        self.speed_violation_buf[env_ids] |= speed_violation
        self.target_motion_invalid_buf[env_ids] |= self.target_motion_manager.invalid_mask[env_ids]
        self.workspace_violation_count[env_ids] += workspace_violation.to(torch.long)
        self.height_violation_count[env_ids] += height_violation.to(torch.long)

    def _workspace_valid(self, env_ids: torch.Tensor) -> torch.Tensor:
        x_min, x_max = self.cfg.workspace_x_range
        y_min, y_max = self.cfg.workspace_y_range
        ego_valid = (self.p_ego_w[env_ids, 0] >= x_min) & (self.p_ego_w[env_ids, 0] <= x_max)
        ego_valid &= (self.p_ego_w[env_ids, 1] >= y_min) & (self.p_ego_w[env_ids, 1] <= y_max)
        target_valid = (self.p_target_w[env_ids, 0] >= x_min) & (self.p_target_w[env_ids, 0] <= x_max)
        target_valid &= (self.p_target_w[env_ids, 1] >= y_min) & (self.p_target_w[env_ids, 1] <= y_max)
        return ego_valid & target_valid

    def _height_valid(self, env_ids: torch.Tensor) -> torch.Tensor:
        z_min, z_max = self.cfg.workspace_z_range
        ego_valid = (self.p_ego_w[env_ids, 2] >= z_min) & (self.p_ego_w[env_ids, 2] <= z_max)
        target_valid = (self.p_target_w[env_ids, 2] >= z_min) & (self.p_target_w[env_ids, 2] <= z_max)
        return ego_valid & target_valid

    def _update_success_diagnostics(self) -> None:
        center_distance = torch.linalg.norm(self.p_rel_w, dim=1)
        offset_error = torch.linalg.norm(self.e_offset_w, dim=1)
        relative_speed = torch.linalg.norm(self.v_ego_w - self.v_target_w, dim=1)
        self._success_step_buf[:] = (
            (offset_error < self._reward_cfg.success_offset_error)
            & (relative_speed < self._reward_cfg.success_relative_speed)
            & (center_distance >= self.cfg.d_safe)
        )
        newly_converged = torch.isnan(self.convergence_time) & self._success_step_buf
        self.convergence_time[newly_converged] = self.episode_length_buf[newly_converged].to(torch.float32) * self.step_dt
        self.success_hold_step_count[:] = torch.where(
            self._success_step_buf,
            self.success_hold_step_count + 1,
            torch.zeros_like(self.success_hold_step_count),
        )
        newly_completed = (self.success_hold_step_count >= self._success_hold_steps_required) & ~self.success_hold_completed_buf
        self._success_completed_step_buf[:] = newly_completed
        self.success_offset_error[newly_completed] = offset_error[newly_completed]
        self.success_relative_speed[newly_completed] = relative_speed[newly_completed]
        self.success_hold_completed_buf[:] |= newly_completed
        self.success_buf[:] |= self.success_hold_completed_buf

    def _target_motion_current_params(self) -> torch.Tensor:
        split_cfg = get_split_cfg(self.cfg.target_motion, self.cfg.target_motion_split)
        max_turn_omega = max(abs(split_cfg.turn_omega_range[0]), abs(split_cfg.turn_omega_range[1]), 1.0e-6)
        return encode_target_motion_current_params(
            self.target_motion_manager.mode_id,
            self.target_motion_manager.v0_w,
            self.target_motion_manager.constant_acceleration_w,
            self.target_motion_manager.turn_omega,
            self.target_motion_manager.current_acceleration_w,
            self.cfg.target_motion.max_speed,
            self.cfg.target_motion.max_acceleration,
            max_turn_omega,
        )

    def _record_episode_log(self, env_ids: torch.Tensor) -> None:
        finished_mask = self.reset_counts[env_ids] > 0
        if not bool(torch.any(finished_mask).item()):
            return
        finished_env_ids = env_ids[finished_mask]
        self._record_episode_metrics(finished_env_ids)

        extras = {}
        for key, episode_sum in self._episode_sums.items():
            episodic_sum_avg = torch.mean(episode_sum[finished_env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
        extras.update(
            {
                "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[finished_env_ids]).item(),
                "Episode_Termination/collision_risk": torch.count_nonzero(
                    self.collision_risk_buf[finished_env_ids]
                ).item(),
                "Episode_Termination/workspace_violation": torch.count_nonzero(
                    self.workspace_violation_buf[finished_env_ids]
                ).item(),
                "Episode_Termination/height_violation": torch.count_nonzero(
                    self.height_violation_buf[finished_env_ids]
                ).item(),
                "Episode_Termination/speed_violation": torch.count_nonzero(
                    self.speed_violation_buf[finished_env_ids]
                ).item(),
                "Episode_Termination/attitude_violation": torch.count_nonzero(
                    self.attitude_violation_buf[finished_env_ids]
                ).item(),
                "Episode_Termination/nan_or_inf": torch.count_nonzero(self.nan_or_inf_buf[finished_env_ids]).item(),
                "Episode_Termination/target_motion_invalid": torch.count_nonzero(
                    self.target_motion_invalid_buf[finished_env_ids]
                ).item(),
                "Metrics/success_rate": torch.mean(self.success_buf[finished_env_ids].to(torch.float32)).item(),
                "Metrics/final_offset_error": torch.mean(self.final_offset_error[finished_env_ids]).item(),
                "Metrics/minimum_center_distance": torch.mean(self.minimum_center_distance[finished_env_ids]).item(),
            }
        )
        self.extras["log"] = extras

    def _record_episode_metrics(self, env_ids: torch.Tensor) -> None:
        action_fraction = self._policy_saturation_fraction(self.action_saturation_count, env_ids)
        acceleration_fraction = self._saturation_fraction(self.acceleration_saturation_count, env_ids)
        speed_fraction = self._saturation_fraction(self.speed_limit_count, env_ids)
        reward_sum = torch.zeros(env_ids.shape[0], dtype=torch.float32, device=self.device)
        for episode_sum in self._episode_sums.values():
            reward_sum += episode_sum[env_ids]
        for index, env_id in enumerate(env_ids.detach().cpu().tolist()):
            self._episode_history.append(
                {
                    "env_id": int(env_id),
                    "initial_offset_error": float(self.initial_offset_error[env_id].item()),
                    "final_offset_error": float(self.final_offset_error[env_id].item()),
                    "minimum_offset_error": float(self.minimum_offset_error[env_id].item()),
                    "final_relative_speed": float(self.final_relative_speed[env_id].item()),
                    "minimum_center_distance": float(self.minimum_center_distance[env_id].item()),
                    "convergence_time": float(self.convergence_time[env_id].item()),
                    "success_offset_error": float(self.success_offset_error[env_id].item()),
                    "success_relative_speed": float(self.success_relative_speed[env_id].item()),
                    "episode_length_steps": int(self.episode_length_buf[env_id].item()),
                    "episode_length_s": float(self.episode_length_buf[env_id].item() * self.step_dt),
                    "success": bool(self.success_buf[env_id].item()),
                    "success_hold_completed": bool(self.success_hold_completed_buf[env_id].item()),
                    "collision_risk_count": int(self.collision_risk_count[env_id].item()),
                    "workspace_violation_count": int(self.workspace_violation_count[env_id].item()),
                    "height_violation_count": int(self.height_violation_count[env_id].item()),
                    "speed_limit_count": int(self.speed_limit_count[env_id].item()),
                    "action_saturation_fraction": float(action_fraction[index].item()),
                    "acceleration_saturation_fraction": float(acceleration_fraction[index].item()),
                    "speed_saturation_fraction": float(speed_fraction[index].item()),
                    "episode_reward_sum": float(reward_sum[index].item()),
                    "theta_offset": float(self.theta_offset[env_id].item()),
                    "delta_radial": float(self.delta_radial[env_id].item()),
                    "delta_tangent": float(self.delta_tangent[env_id].item()),
                }
            )

    def _saturation_fraction(self, count_tensor: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(self.rl_physics_step_count[env_ids].to(torch.float32), min=1.0)
        return count_tensor[env_ids].to(torch.float32) / denominator

    def _policy_saturation_fraction(self, count_tensor: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(self.rl_policy_step_count[env_ids].to(torch.float32), min=1.0)
        return count_tensor[env_ids].to(torch.float32) / denominator

    def get_m5_episode_history(self, clear: bool = False) -> list[dict[str, float | int | bool]]:
        history = list(self._episode_history)
        if clear:
            self._episode_history.clear()
        return history

    def get_m5_diagnostics(self) -> dict[str, object]:
        relative_speed = torch.linalg.norm(self.v_ego_w - self.v_target_w, dim=1)
        offset_error = torch.linalg.norm(self.e_offset_w, dim=1)
        center_distance = torch.linalg.norm(self.p_rel_w, dim=1)
        policy_obs, critic_obs = self._get_observations().values()
        return {
            "num_envs": int(self.num_envs),
            "total_steps": int(self.common_step_counter),
            "sim_dt": float(self.physics_dt),
            "decimation": int(self.cfg.decimation),
            "step_dt": float(self.step_dt),
            "policy_obs_dim": int(policy_obs.shape[1]),
            "critic_obs_dim": int(critic_obs.shape[1]),
            "offset_error": offset_error.detach().cpu().tolist(),
            "relative_speed": relative_speed.detach().cpu().tolist(),
            "center_distance_min": float(center_distance.min().item()),
            "success": self.success_buf.detach().cpu().tolist(),
            "success_hold_completed": self.success_hold_completed_buf.detach().cpu().tolist(),
            "collision_risk_count": self.collision_risk_count.detach().cpu().tolist(),
            "workspace_violation_count": self.workspace_violation_count.detach().cpu().tolist(),
            "height_violation_count": self.height_violation_count.detach().cpu().tolist(),
            "speed_limit_count": self.speed_limit_count.detach().cpu().tolist(),
            "action_saturation_fraction": self._policy_saturation_fraction(
                self.action_saturation_count, self._all_env_ids
            ).detach().cpu().tolist(),
            "acceleration_saturation_fraction": self._saturation_fraction(
                self.acceleration_saturation_count, self._all_env_ids
            ).detach().cpu().tolist(),
            "speed_saturation_fraction": self._saturation_fraction(
                self.speed_limit_count, self._all_env_ids
            ).detach().cpu().tolist(),
            "raw_action_abs_max": float(torch.abs(self.raw_action).max().item()),
            "squashed_action_abs_max": float(torch.abs(self.squashed_action).max().item()),
            "v_cmd_abs_max": float(torch.abs(self.v_cmd_w).max().item()),
            "ego_speed_max_observed": float(self._ego_speed_max_observed.max().item()),
            "ego_acceleration_max_observed": float(self._ego_acceleration_max_observed.max().item()),
            "b_des_norm_min": float(torch.linalg.norm(self.b_des_w, dim=1).min().item()),
            "b_des_norm_max": float(torch.linalg.norm(self.b_des_w, dim=1).max().item()),
            "finite_check": all_finite(
                self.p_ego_w,
                self.v_ego_w,
                self.p_target_w,
                self.v_target_w,
                self.p_rel_w,
                self.v_rel_w,
                self.e_offset_w,
                self.raw_action,
                self.squashed_action,
                self.v_cmd_w,
                self.a_cmd_w,
                self.b_des_w,
                policy_obs,
                critic_obs,
            ),
        }
