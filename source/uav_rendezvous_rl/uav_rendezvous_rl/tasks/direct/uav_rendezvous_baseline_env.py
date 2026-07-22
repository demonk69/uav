"""Independent M4 deterministic non-learning baseline DirectRLEnv task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv

from uav_rendezvous_rl.controllers import (
    compute_baseline_velocity_command,
    compute_limited_acceleration,
    integrate_ego_kinematics,
    sample_baseline_initial_ego_state,
    validate_baseline_initial_geometry,
)
from uav_rendezvous_rl.motions.configs import get_split_cfg
from uav_rendezvous_rl.motions.sampling import uniform_range

from .m2_kinematics import all_finite
from .m4_accounting import update_collision_risk_accounting
from .uav_rendezvous_baseline_env_cfg import UavRendezvousBaselineEnvCfg
from .uav_rendezvous_env import UavRendezvousEnv


class UavRendezvousBaselineEnv(UavRendezvousEnv):
    """M4 deterministic baseline with moving ego and current-state target prediction."""

    cfg: UavRendezvousBaselineEnvCfg

    def __init__(self, cfg: UavRendezvousBaselineEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._baseline_cfg = self.cfg.baseline_controller
        self._initial_geometry_cfg = self.cfg.baseline_initial_geometry
        self._success_hold_steps_required = max(1, int(round(self._baseline_cfg.success_hold_s / self.physics_dt)))

        self.p_target_pred_w = torch.zeros_like(self.p_ego_w)
        self.p_goal_w = torch.zeros_like(self.p_ego_w)
        self.v_cmd_w = torch.zeros_like(self.p_ego_w)
        self.a_cmd_w = torch.zeros_like(self.p_ego_w)
        self.delta_initial_w = torch.zeros_like(self.p_ego_w)
        self.initial_path_min_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.workspace_violation_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.speed_violation_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nan_or_inf_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_hold_completed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_hold_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.initial_offset_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.final_offset_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.minimum_offset_error = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self.final_relative_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.minimum_center_distance = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        self.convergence_time = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)

        self.collision_risk_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.workspace_violation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.speed_limit_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.acceleration_saturation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.velocity_command_saturation_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.baseline_physics_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._ego_speed_max_observed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ego_acceleration_max_observed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_history: list[dict[str, float | int | bool]] = []

    def _apply_action(self) -> None:
        # Causality: control uses target state at the start of this physics substep.
        baseline_cfg = self._baseline_cfg
        self.v_cmd_w[:], v_cmd_saturated, self.p_target_pred_w[:], self.p_goal_w[:] = compute_baseline_velocity_command(
            self.p_ego_w,
            self.p_target_w,
            self.v_target_w,
            self.b_des_w,
            baseline_cfg.prediction_horizon_s,
            baseline_cfg.v_cmd_max,
        )
        self.a_cmd_w[:], acceleration_saturated = compute_limited_acceleration(
            self.v_cmd_w, self.v_ego_w, baseline_cfg.tau_v, baseline_cfg.a_max
        )
        p_ego_next_w, v_ego_next_w, speed_saturated = integrate_ego_kinematics(
            self.p_ego_w, self.v_ego_w, self.a_cmd_w, self.physics_dt, baseline_cfg.v_abs_max
        )

        self.p_ego_w[:] = p_ego_next_w
        self.v_ego_w[:] = v_ego_next_w
        self._check_collision_against_current_target()

        target_state = self.target_motion_manager.step()
        self.p_target_w[:] = target_state.p_target_w
        self.v_target_w[:] = target_state.v_target_w
        self.a_target_w[:] = target_state.a_target_w
        self.target_elapsed_time[:] = self.motion_step_count.to(dtype=torch.float32) * self.physics_dt

        self.baseline_physics_step_count += 1
        self.velocity_command_saturation_count += v_cmd_saturated.to(torch.long)
        self.acceleration_saturation_count += acceleration_saturated.to(torch.long)
        self.speed_limit_count += speed_saturated.to(torch.long)
        self._ego_speed_max_observed[:] = torch.maximum(self._ego_speed_max_observed, torch.linalg.norm(self.v_ego_w, dim=1))
        self._ego_acceleration_max_observed[:] = torch.maximum(
            self._ego_acceleration_max_observed, torch.linalg.norm(self.a_cmd_w, dim=1)
        )

        self._refresh_relative_state()
        self._update_baseline_diagnostics()
        self._update_motion_diagnostics()
        self._write_entities_to_sim(self._all_env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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
                self.v_cmd_w,
                self.a_cmd_w,
            ),
            dim=-1,
        )
        self.nan_or_inf_buf[:] = torch.any(~torch.isfinite(state_tensors), dim=1)
        terminated = (
            self.collision_risk_buf
            | self.workspace_violation_buf
            | self.speed_violation_buf
            | self.nan_or_inf_buf
            | self.target_motion_invalid_buf
        )
        if self._baseline_cfg.terminate_on_success:
            terminated |= self.success_hold_completed_buf
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if hasattr(self, "reset_counts"):
            self._record_episode_metrics(env_ids)
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

        b_des_reset_w = self.b_des_w[env_ids]
        p_ego_initial_w, v_ego_initial_w, delta_initial_w = sample_baseline_initial_ego_state(
            p_target_initial_w, b_des_reset_w, self._initial_geometry_cfg, self._rng, self.device
        )
        geometry = validate_baseline_initial_geometry(
            p_target_initial_w,
            p_ego_initial_w,
            b_des_reset_w,
            self.cfg.d_safe,
            self._initial_geometry_cfg.min_initial_offset_error,
        )
        if not bool(torch.all(geometry["valid"]).item()):
            raise RuntimeError("M4 baseline reset produced invalid initial geometry.")

        target_state = self.target_motion_manager.reset(
            env_ids, p_target_initial_w, v_target_initial_w, self._rng, self.cfg.target_motion_split
        )

        self._actions[env_ids] = 0.0
        self.p_ego_w[env_ids] = p_ego_initial_w
        self.v_ego_w[env_ids] = v_ego_initial_w
        self.p_target_initial_w[env_ids] = p_target_initial_w
        self.p_target_w[env_ids] = target_state.p_target_w[env_ids]
        self.v_target_w[env_ids] = target_state.v_target_w[env_ids]
        self.a_target_w[env_ids] = target_state.a_target_w[env_ids]
        self.delta_initial_w[env_ids] = delta_initial_w
        self.initial_path_min_distance[env_ids] = geometry["segment_distance"]
        self.target_elapsed_time[env_ids] = 0.0
        self.target_motion_invalid_buf[env_ids] = self.target_motion_manager.invalid_mask[env_ids]

        self._refresh_relative_state()
        offset_error = torch.linalg.norm(self.e_offset_w[env_ids], dim=1)
        center_distance = torch.linalg.norm(self.p_rel_w[env_ids], dim=1)
        relative_speed = torch.linalg.norm(self.v_ego_w[env_ids] - self.v_target_w[env_ids], dim=1)

        self.p_target_pred_w[env_ids] = self.p_target_w[env_ids]
        self.p_goal_w[env_ids] = self.p_target_w[env_ids] + self.b_des_w[env_ids]
        self.v_cmd_w[env_ids] = 0.0
        self.a_cmd_w[env_ids] = 0.0
        self.collision_risk_buf[env_ids] = False
        self.workspace_violation_buf[env_ids] = False
        self.speed_violation_buf[env_ids] = False
        self.nan_or_inf_buf[env_ids] = False
        self.success_buf[env_ids] = False
        self.success_hold_completed_buf[env_ids] = False
        self.success_hold_step_count[env_ids] = 0
        self.initial_offset_error[env_ids] = offset_error
        self.final_offset_error[env_ids] = offset_error
        self.minimum_offset_error[env_ids] = offset_error
        self.final_relative_speed[env_ids] = relative_speed
        self.minimum_center_distance[env_ids] = center_distance
        self.min_relative_distance_per_episode[env_ids] = center_distance
        self.convergence_time[env_ids] = float("nan")
        self.collision_risk_count[env_ids] = 0
        self.workspace_violation_count[env_ids] = 0
        self.speed_limit_count[env_ids] = 0
        self.acceleration_saturation_count[env_ids] = 0
        self.velocity_command_saturation_count[env_ids] = 0
        self.baseline_physics_step_count[env_ids] = 0
        self._ego_speed_max_observed[env_ids] = torch.linalg.norm(self.v_ego_w[env_ids], dim=1)
        self._ego_acceleration_max_observed[env_ids] = 0.0
        self._workspace_abs_max_observed[env_ids] = torch.maximum(
            torch.abs(self.p_ego_w[env_ids]).amax(dim=1), torch.abs(self.p_target_w[env_ids]).amax(dim=1)
        )
        self._target_speed_max_observed[env_ids] = torch.linalg.norm(self.v_target_w[env_ids], dim=1)
        self._target_acceleration_max_observed[env_ids] = torch.linalg.norm(self.a_target_w[env_ids], dim=1)
        self.reset_counts[env_ids] += 1
        self._write_entities_to_sim(env_ids)

    def _check_collision_against_current_target(self) -> None:
        center_distance = torch.linalg.norm(self.p_target_w - self.p_ego_w, dim=1)
        collision = center_distance < self.cfg.d_safe
        update_collision_risk_accounting(collision, self.collision_risk_buf, self.collision_risk_count)
        self.minimum_center_distance[:] = torch.minimum(self.minimum_center_distance, center_distance)
        self.min_relative_distance_per_episode[:] = torch.minimum(self.min_relative_distance_per_episode, center_distance)
        self._min_relative_distance_observed[:] = torch.minimum(self._min_relative_distance_observed, center_distance)

    def _update_baseline_diagnostics(self) -> None:
        center_distance = torch.linalg.norm(self.p_rel_w, dim=1)
        offset_error = torch.linalg.norm(self.e_offset_w, dim=1)
        relative_speed = torch.linalg.norm(self.v_ego_w - self.v_target_w, dim=1)
        workspace = self._workspace_valid()
        speed = torch.linalg.norm(self.v_ego_w, dim=1)
        speed_violation = speed > self._baseline_cfg.v_abs_max + 1.0e-5
        collision = center_distance < self.cfg.d_safe

        self.final_offset_error[:] = offset_error
        self.minimum_offset_error[:] = torch.minimum(self.minimum_offset_error, offset_error)
        self.final_relative_speed[:] = relative_speed
        self.minimum_center_distance[:] = torch.minimum(self.minimum_center_distance, center_distance)
        self.min_relative_distance_per_episode[:] = torch.minimum(self.min_relative_distance_per_episode, center_distance)
        self._min_relative_distance_observed[:] = torch.minimum(self._min_relative_distance_observed, center_distance)
        update_collision_risk_accounting(collision, self.collision_risk_buf, self.collision_risk_count)
        self.workspace_violation_buf[:] |= ~workspace
        self.speed_violation_buf[:] |= speed_violation
        self.target_motion_invalid_buf[:] |= self.target_motion_manager.invalid_mask
        self.workspace_violation_count += (~workspace).to(torch.long)

        success_step = (
            (offset_error < self._baseline_cfg.success_offset_error)
            & (relative_speed < self._baseline_cfg.success_relative_speed)
            & (center_distance >= self.cfg.d_safe)
        )
        self.success_hold_step_count[:] = torch.where(
            success_step, self.success_hold_step_count + 1, torch.zeros_like(self.success_hold_step_count)
        )
        newly_converged = (self.convergence_time != self.convergence_time) & success_step
        self.convergence_time[newly_converged] = self.baseline_physics_step_count[newly_converged].to(torch.float32)
        self.convergence_time[newly_converged] *= self.physics_dt
        self.success_hold_completed_buf[:] |= self.success_hold_step_count >= self._success_hold_steps_required
        self.success_buf[:] |= self.success_hold_completed_buf

    def _workspace_valid(self) -> torch.Tensor:
        x_min, x_max = self.cfg.workspace_x_range
        y_min, y_max = self.cfg.workspace_y_range
        z_min, z_max = self.cfg.workspace_z_range
        ego_valid = (self.p_ego_w[:, 0] >= x_min) & (self.p_ego_w[:, 0] <= x_max)
        ego_valid &= (self.p_ego_w[:, 1] >= y_min) & (self.p_ego_w[:, 1] <= y_max)
        ego_valid &= (self.p_ego_w[:, 2] >= z_min) & (self.p_ego_w[:, 2] <= z_max)
        target_valid = (self.p_target_w[:, 0] >= x_min) & (self.p_target_w[:, 0] <= x_max)
        target_valid &= (self.p_target_w[:, 1] >= y_min) & (self.p_target_w[:, 1] <= y_max)
        target_valid &= (self.p_target_w[:, 2] >= z_min) & (self.p_target_w[:, 2] <= z_max)
        return ego_valid & target_valid

    def _record_episode_metrics(self, env_ids: torch.Tensor) -> None:
        finished_mask = self.reset_counts[env_ids] > 0
        if not bool(torch.any(finished_mask).item()):
            return
        finished_env_ids = env_ids[finished_mask]
        acceleration_fraction = self._saturation_fraction(self.acceleration_saturation_count, finished_env_ids)
        velocity_fraction = self._saturation_fraction(self.velocity_command_saturation_count, finished_env_ids)
        for index, env_id in enumerate(finished_env_ids.detach().cpu().tolist()):
            self._episode_history.append(
                {
                    "env_id": int(env_id),
                    "initial_offset_error": float(self.initial_offset_error[env_id].item()),
                    "final_offset_error": float(self.final_offset_error[env_id].item()),
                    "minimum_offset_error": float(self.minimum_offset_error[env_id].item()),
                    "final_relative_speed": float(self.final_relative_speed[env_id].item()),
                    "minimum_center_distance": float(self.minimum_center_distance[env_id].item()),
                    "convergence_time": float(self.convergence_time[env_id].item()),
                    "success": bool(self.success_buf[env_id].item()),
                    "success_hold_completed": bool(self.success_hold_completed_buf[env_id].item()),
                    "collision_risk_count": int(self.collision_risk_count[env_id].item()),
                    "workspace_violation_count": int(self.workspace_violation_count[env_id].item()),
                    "speed_limit_count": int(self.speed_limit_count[env_id].item()),
                    "acceleration_saturation_fraction": float(acceleration_fraction[index].item()),
                    "velocity_command_saturation_fraction": float(velocity_fraction[index].item()),
                    "initial_path_min_distance": float(self.initial_path_min_distance[env_id].item()),
                }
            )

    def _saturation_fraction(self, count_tensor: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(self.baseline_physics_step_count[env_ids].to(torch.float32), min=1.0)
        return count_tensor[env_ids].to(torch.float32) / denominator

    def get_m4_episode_history(self, clear: bool = False) -> list[dict[str, float | int | bool]]:
        history = list(self._episode_history)
        if clear:
            self._episode_history.clear()
        return history

    def get_m4_diagnostics(self) -> dict[str, object]:
        relative_speed = torch.linalg.norm(self.v_ego_w - self.v_target_w, dim=1)
        offset_error = torch.linalg.norm(self.e_offset_w, dim=1)
        return {
            "num_envs": int(self.num_envs),
            "total_steps": int(self.common_step_counter),
            "sim_dt": float(self.physics_dt),
            "decimation": int(self.cfg.decimation),
            "step_dt": float(self.step_dt),
            "offset_error": offset_error.detach().cpu().tolist(),
            "relative_speed": relative_speed.detach().cpu().tolist(),
            "success": self.success_buf.detach().cpu().tolist(),
            "success_hold_completed": self.success_hold_completed_buf.detach().cpu().tolist(),
            "minimum_center_distance": self.minimum_center_distance.detach().cpu().tolist(),
            "collision_risk_count": self.collision_risk_count.detach().cpu().tolist(),
            "workspace_violation_count": self.workspace_violation_count.detach().cpu().tolist(),
            "speed_limit_count": self.speed_limit_count.detach().cpu().tolist(),
            "acceleration_saturation_fraction": self._saturation_fraction(
                self.acceleration_saturation_count, self._all_env_ids
            ).detach().cpu().tolist(),
            "velocity_command_saturation_fraction": self._saturation_fraction(
                self.velocity_command_saturation_count, self._all_env_ids
            ).detach().cpu().tolist(),
            "ego_speed_max_observed": float(self._ego_speed_max_observed.max().item()),
            "ego_acceleration_max_observed": float(self._ego_acceleration_max_observed.max().item()),
            "finite_check": all_finite(
                self.p_ego_w,
                self.v_ego_w,
                self.p_target_w,
                self.v_target_w,
                self.p_rel_w,
                self.v_rel_w,
                self.e_offset_w,
                self.v_cmd_w,
                self.a_cmd_w,
            ),
        }
