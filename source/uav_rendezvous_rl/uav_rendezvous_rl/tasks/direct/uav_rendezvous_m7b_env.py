"""Independent M7B dynamics-robustness DirectRLEnv task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from uav_rendezvous_rl.dynamics import (
    ActionDelayBuffer,
    WindGustState,
    integrate_m7b_dynamics,
    sample_steady_wind,
    sample_gust_segment,
    stateless_m7b_randint,
    stateless_m7b_uniform,
    step_wind_and_gust,
    validate_m7b_dynamics_config,
)
from uav_rendezvous_rl.mdp import assemble_actor_observation, assemble_critic_observation_m7b
from uav_rendezvous_rl.mdp.rendezvous import map_raw_action_to_velocity_command
from uav_rendezvous_rl.observations import ObservationPipeline

from .m2_kinematics import all_finite
from .uav_rendezvous_m7b_env_cfg import UavRendezvousM7BEnvCfg
from .uav_rendezvous_recurrent_env import UavRendezvousRecurrentEnv


class UavRendezvousM7BEnv(UavRendezvousRecurrentEnv):
    """M7B task with dynamics randomization, action delay, and wind."""

    cfg: UavRendezvousM7BEnvCfg

    def __init__(self, cfg: UavRendezvousM7BEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        validate_m7b_dynamics_config(cfg.m7b_dynamics)
        self._m7b_seed = int(cfg.seed if cfg.seed is not None else 0)
        self._m7b_device = self.device
        self._m7b_dtype = torch.float32
        self._num_envs = int(self.num_envs)

        self.observation_pipeline = ObservationPipeline(
            cfg.observation_degradation, self._num_envs, self._m7b_device, seed=self._m7b_seed
        )
        self.p_rel_obs_w = torch.zeros_like(self.p_rel_w)
        self.v_rel_obs_w = torch.zeros_like(self.v_rel_w)

        self.action_delay_buffer = ActionDelayBuffer(self._num_envs, int(cfg.m7b_dynamics.max_delay_steps), self._m7b_device, self._m7b_dtype)
        self.executed_squashed_action = torch.zeros((self._num_envs, 3), dtype=self._m7b_dtype, device=self._m7b_device)
        self._m7b_tau_velocity_scale = torch.ones(self._num_envs, dtype=self._m7b_dtype, device=self._m7b_device)
        self._m7b_acceleration_limit_scale = torch.ones(self._num_envs, dtype=self._m7b_dtype, device=self._m7b_device)
        self._m7b_speed_limit_scale = torch.ones(self._num_envs, dtype=self._m7b_dtype, device=self._m7b_device)
        self._m7b_linear_drag = torch.zeros(self._num_envs, dtype=self._m7b_dtype, device=self._m7b_device)
        self._m7b_action_delay_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._m7b_device)
        self._m7b_episode_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._m7b_device)
        self._m7b_total_safety_saturated_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._m7b_device)
        self._total_safety_saturated_step = torch.zeros(self._num_envs, dtype=torch.bool, device=self._m7b_device)
        self._m7b_decrement_gust_this_substep = False

        self._wind_state = WindGustState(
            steady_wind_w=torch.zeros((self._num_envs, 3), dtype=self._m7b_dtype, device=self._m7b_device),
            gust_target_w=torch.zeros((self._num_envs, 3), dtype=self._m7b_dtype, device=self._m7b_device),
            gust_current_w=torch.zeros((self._num_envs, 3), dtype=self._m7b_dtype, device=self._m7b_device),
            gust_remaining_policy_steps=torch.zeros(self._num_envs, dtype=torch.long, device=self._m7b_device),
            gust_segment_index=torch.zeros(self._num_envs, dtype=torch.long, device=self._m7b_device),
        )

        self._m7b_sample_episode_params(self._all_env_ids)
        self._m7b_wind_reset(self._all_env_ids)
        self._cached_wind_accel_w = self._wind_state.steady_wind_w + self._wind_state.gust_current_w
        self.action_delay_buffer.reset(self._all_env_ids)
        self.observation_pipeline.reset(self._all_env_ids, self.p_rel_w, self.v_rel_w)
        self.p_rel_obs_w[:] = self.p_rel_w
        self.v_rel_obs_w[:] = self.v_rel_w

    def _m7b_sample_episode_params(self, env_ids: torch.Tensor) -> None:
        eps = self._m7b_episode_count[env_ids]
        seed = self._m7b_seed
        self._m7b_tau_velocity_scale[env_ids] = self._sample_scale_or_nominal(
            env_ids, eps, seed, 1001, 0.75, float(self.cfg.tau_velocity_scale)
        )
        self._m7b_acceleration_limit_scale[env_ids] = self._sample_scale_or_nominal(
            env_ids, eps, seed, 1002, 0.80, float(self.cfg.acceleration_limit_scale)
        )
        self._m7b_speed_limit_scale[env_ids] = self._sample_scale_or_nominal(
            env_ids, eps, seed, 1003, 0.90, float(self.cfg.speed_limit_scale)
        )
        self._m7b_linear_drag[env_ids] = self._sample_zero_to_max(env_ids, eps, seed, 1004, float(self.cfg.linear_drag))
        if int(self.cfg.action_delay_steps) > 0:
            self._m7b_action_delay_steps[env_ids] = stateless_m7b_randint(
                env_ids, eps, 0, int(self.cfg.action_delay_steps), seed, 1005, 0
            )
        else:
            self._m7b_action_delay_steps[env_ids] = 0

    def _sample_scale_or_nominal(
        self, env_ids: torch.Tensor, episode_counts: torch.Tensor, seed: int, stream: int, low: float, high: float
    ) -> torch.Tensor:
        if high <= 1.0:
            return torch.ones(env_ids.numel(), dtype=self._m7b_dtype, device=self._m7b_device)
        samples = stateless_m7b_uniform(env_ids, episode_counts, seed, stream, 0).squeeze(-1)
        return float(low) + samples * (float(high) - float(low))

    def _sample_zero_to_max(
        self, env_ids: torch.Tensor, episode_counts: torch.Tensor, seed: int, stream: int, high: float
    ) -> torch.Tensor:
        if high <= 0.0:
            return torch.zeros(env_ids.numel(), dtype=self._m7b_dtype, device=self._m7b_device)
        return stateless_m7b_uniform(env_ids, episode_counts, seed, stream, 0).squeeze(-1) * float(high)

    def _m7b_wind_reset(self, env_ids: torch.Tensor) -> None:
        eps = self._m7b_episode_count[env_ids]
        seed = self._m7b_seed
        cfg = self.cfg.m7b_dynamics
        self._wind_state.gust_segment_index[env_ids] = 0
        self._wind_state.steady_wind_w[env_ids] = sample_steady_wind(
            env_ids, eps, seed, self.cfg.steady_wind_max_magnitude, cfg.a_max_nominal, self._m7b_device, self._m7b_dtype
        )
        gust_target, gust_duration = sample_gust_segment(
            env_ids, eps, seed, self.cfg.gust_max_magnitude, cfg.gust_duration_min, cfg.gust_duration_max,
            cfg.a_max_nominal, self._m7b_device, self._m7b_dtype, sample_counter=0,
        )
        self._wind_state.gust_target_w[env_ids] = gust_target
        self._wind_state.gust_current_w[env_ids] = gust_target
        self._wind_state.gust_remaining_policy_steps[env_ids] = gust_duration
        self._m7b_total_safety_saturated_count[env_ids] = 0

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        resolved_env_ids = self._resolve_env_ids(env_ids)
        super()._reset_idx(resolved_env_ids)
        self._m7b_episode_count[resolved_env_ids] += 1
        self._m7b_sample_episode_params(resolved_env_ids)
        self._m7b_wind_reset(resolved_env_ids)
        self._cached_wind_accel_w[resolved_env_ids] = (
            self._wind_state.steady_wind_w[resolved_env_ids] + self._wind_state.gust_current_w[resolved_env_ids]
        )
        self.action_delay_buffer.reset(resolved_env_ids)
        if hasattr(self, "observation_pipeline"):
            self.observation_pipeline.reset(resolved_env_ids, self.p_rel_w[resolved_env_ids], self.v_rel_w[resolved_env_ids])
            self.p_rel_obs_w[resolved_env_ids] = self.p_rel_w[resolved_env_ids]
            self.v_rel_obs_w[resolved_env_ids] = self.v_rel_w[resolved_env_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 3:
            raise RuntimeError(f"M7B action must have shape (num_envs, 3), got {tuple(actions.shape)}.")
        non_finite = ~torch.isfinite(actions).all(dim=1)
        self.nan_or_inf_buf[:] |= non_finite
        self.raw_action[:] = torch.nan_to_num(actions.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        squashed_action, v_cmd_w = map_raw_action_to_velocity_command(self.raw_action, self.cfg.m7b_dynamics.v_max)
        self._action_delta_squashed[:] = squashed_action - self.previous_squashed_action
        self.squashed_action[:] = squashed_action
        self.previous_squashed_action[:] = squashed_action

        self.executed_squashed_action[:] = self.action_delay_buffer.push_and_read(squashed_action, self._m7b_action_delay_steps)
        v_cmd_delayed_w = self.cfg.m7b_dynamics.v_max * self.executed_squashed_action
        self.v_cmd_w[:] = v_cmd_delayed_w

        self.action_saturation_count += torch.any(torch.abs(squashed_action) >= 0.95, dim=1).to(torch.long)
        self.rl_policy_step_count += 1
        self._acceleration_saturated_step[:] = False
        self._speed_saturated_step[:] = False
        self._total_safety_saturated_step[:] = False

        expired = self._wind_state.gust_remaining_policy_steps <= 0
        if expired.any():
            expired_env_ids = torch.nonzero(expired, as_tuple=False).squeeze(-1)
            self._wind_state.gust_segment_index[expired_env_ids] += 1
            eps = self._m7b_episode_count[expired_env_ids]
            cfg = self.cfg.m7b_dynamics
            new_target, new_duration = sample_gust_segment(
                expired_env_ids, eps, self._m7b_seed, self.cfg.gust_max_magnitude,
                cfg.gust_duration_min, cfg.gust_duration_max, cfg.a_max_nominal, self._m7b_device, self._m7b_dtype,
                sample_counter=self._wind_state.gust_segment_index[expired_env_ids],
            )
            self._wind_state.gust_target_w[expired_env_ids] = new_target
            self._wind_state.gust_remaining_policy_steps[expired_env_ids] = new_duration
        self._m7b_decrement_gust_this_substep = True

    def _apply_action(self) -> None:
        tau_vel = self.cfg.m7b_dynamics.tau_v_nominal * self._m7b_tau_velocity_scale
        acc_lim = self.cfg.m7b_dynamics.a_max_nominal * self._m7b_acceleration_limit_scale
        spd_lim = self.cfg.m7b_dynamics.v_abs_max_nominal * self._m7b_speed_limit_scale
        wind_accel = step_wind_and_gust(
            self._wind_state,
            self.physics_dt,
            self.cfg.m7b_dynamics.gust_tau_s,
            self._m7b_decrement_gust_this_substep,
        )
        self._m7b_decrement_gust_this_substep = False
        self._cached_wind_accel_w = wind_accel

        p_next, v_next, track_sat, total_sat, speed_sat = integrate_m7b_dynamics(
            self.p_ego_w, self.v_ego_w, self.v_cmd_w,
            wind_accel,
            self.physics_dt, tau_vel, acc_lim, spd_lim, self._m7b_linear_drag,
            self.cfg.m7b_dynamics.a_applied_abs_max,
        )
        observed_accel = (v_next - self.v_ego_w) / float(self.physics_dt)
        self.p_ego_w[:] = p_next
        self.v_ego_w[:] = v_next
        self._check_collision_against_current_target()

        target_state = self.target_motion_manager.step()
        self.p_target_w[:] = target_state.p_target_w
        self.v_target_w[:] = target_state.v_target_w
        self.a_target_w[:] = target_state.a_target_w
        self.target_elapsed_time[:] = self.motion_step_count.to(dtype=torch.float32) * self.physics_dt

        self._acceleration_saturated_step[:] |= track_sat
        self._total_safety_saturated_step[:] |= total_sat
        self._speed_saturated_step[:] |= speed_sat
        self.acceleration_saturation_count += track_sat.to(torch.long)
        self._m7b_total_safety_saturated_count += total_sat.to(torch.long)
        self.speed_limit_count += speed_sat.to(torch.long)
        self.rl_physics_step_count += 1
        self._ego_speed_max_observed[:] = torch.maximum(self._ego_speed_max_observed, torch.linalg.norm(self.v_ego_w, dim=1))
        self._ego_acceleration_max_observed[:] = torch.maximum(
            self._ego_acceleration_max_observed, torch.linalg.norm(observed_accel, dim=1)
        )

        self._refresh_relative_state()
        self._update_rl_diagnostics()
        self._update_motion_diagnostics()
        self._write_entities_to_sim(self._all_env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._write_entities_to_sim(self._all_env_ids)
        self.p_rel_obs_w[:], self.v_rel_obs_w[:] = self.observation_pipeline.observe(self.p_rel_w, self.v_rel_w)
        actor_obs = assemble_actor_observation(
            self.p_rel_obs_w, self.v_rel_obs_w, self.v_ego_w,
            self._r_ego_6d, self._omega_ego_b, self.previous_squashed_action,
            self.b_des_w, self.cfg.d_offset,
        )
        mode_one_hot = F.one_hot(self.target_motion_manager.mode_id, num_classes=4).to(dtype=torch.float32)
        target_params = self._target_motion_current_params()
        episode_phase = torch.clamp(self.episode_length_buf.to(dtype=torch.float32) / float(self.max_episode_length), 0.0, 1.0)
        norm_delay = self._m7b_action_delay_steps.to(dtype=torch.float32) / max(1.0, float(self.cfg.m7b_dynamics.max_delay_steps))
        wind = getattr(self, "_cached_wind_accel_w", torch.zeros_like(self.p_ego_w))
        critic_obs = assemble_critic_observation_m7b(
            actor_obs, self.p_ego_w, self.p_target_w, self.v_target_w, self.a_target_w,
            self._r_target_6d, self._omega_target_b, mode_one_hot, target_params, episode_phase,
            self._m7b_tau_velocity_scale, self._m7b_acceleration_limit_scale,
            self._m7b_speed_limit_scale, self._m7b_linear_drag, norm_delay, wind,
        )
        return {"policy": actor_obs, "critic": critic_obs}

    def _update_rl_diagnostics(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self._all_env_ids
        center_distance = torch.linalg.norm(self.p_rel_w[env_ids], dim=1)
        offset_error = torch.linalg.norm(self.e_offset_w[env_ids], dim=1)
        relative_speed = torch.linalg.norm(self.v_ego_w[env_ids] - self.v_target_w[env_ids], dim=1)
        speed = torch.linalg.norm(self.v_ego_w[env_ids], dim=1)
        speed_limit = self.cfg.m7b_dynamics.v_abs_max_nominal * self._m7b_speed_limit_scale[env_ids]
        workspace_valid = self._workspace_valid(env_ids)
        height_valid = self._height_valid(env_ids)
        speed_valid = speed <= speed_limit + 1.0e-5
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

    def _record_episode_metrics(self, env_ids: torch.Tensor) -> None:
        start_index = len(self._episode_history)
        super()._record_episode_metrics(env_ids)
        total_safety_fraction = self._saturation_fraction(self._m7b_total_safety_saturated_count, env_ids)
        for entry, env_id, safety_fraction in zip(
            self._episode_history[start_index:], env_ids.detach().cpu().tolist(), total_safety_fraction.detach().cpu().tolist(), strict=True
        ):
            entry["tau_velocity_scale"] = float(self._m7b_tau_velocity_scale[env_id].item())
            entry["acceleration_limit_scale"] = float(self._m7b_acceleration_limit_scale[env_id].item())
            entry["speed_limit_scale"] = float(self._m7b_speed_limit_scale[env_id].item())
            entry["linear_drag"] = float(self._m7b_linear_drag[env_id].item())
            entry["action_delay_steps"] = int(self._m7b_action_delay_steps[env_id].item())
            entry["total_safety_saturation_fraction"] = float(safety_fraction)

    def get_m7b_diagnostics(self) -> dict[str, object]:
        policy_obs = self.obs_buf.get("policy", None) if isinstance(getattr(self, "obs_buf", None), dict) else None
        critic_obs = self.obs_buf.get("critic", None) if isinstance(getattr(self, "obs_buf", None), dict) else None
        if policy_obs is None or critic_obs is None:
            actor_obs = assemble_actor_observation(
                self.p_rel_obs_w, self.v_rel_obs_w, self.v_ego_w,
                self._r_ego_6d, self._omega_ego_b, self.previous_squashed_action,
                self.b_des_w, self.cfg.d_offset,
            )
            mode_one_hot = F.one_hot(self.target_motion_manager.mode_id, num_classes=4).to(dtype=torch.float32)
            episode_phase = torch.clamp(self.episode_length_buf.to(dtype=torch.float32) / float(self.max_episode_length), 0.0, 1.0)
            norm_delay = self._m7b_action_delay_steps.to(dtype=torch.float32) / max(1.0, float(self.cfg.m7b_dynamics.max_delay_steps))
            wind = getattr(self, "_cached_wind_accel_w", torch.zeros_like(self.p_ego_w))
            critic_obs = assemble_critic_observation_m7b(
                actor_obs, self.p_ego_w, self.p_target_w, self.v_target_w, self.a_target_w,
                self._r_target_6d, self._omega_target_b, mode_one_hot,
                self._target_motion_current_params(), episode_phase,
                self._m7b_tau_velocity_scale, self._m7b_acceleration_limit_scale,
                self._m7b_speed_limit_scale, self._m7b_linear_drag, norm_delay, wind,
            )
            policy_obs = actor_obs
        finite = all_finite(
            self.p_ego_w, self.v_ego_w, self.p_target_w, self.v_target_w, self.a_target_w,
            self.p_rel_w, self.v_rel_w, self.e_offset_w, self.p_rel_obs_w, self.v_rel_obs_w,
            policy_obs, critic_obs,
        )
        wind = getattr(self, "_cached_wind_accel_w", torch.zeros_like(self.p_ego_w))
        return {
            "num_envs": int(self.num_envs),
            "total_steps": int(self.common_step_counter),
            "policy_obs_dim": int(policy_obs.shape[1]),
            "critic_obs_dim": int(critic_obs.shape[1]),
            "action_dim": int(self.single_action_space.shape[0]),
            "finite_check": finite,
            "mode_counts": self.target_motion_manager.mode_counts(),
            "target_motion_split": self.cfg.target_motion_split,
            "observation_pipeline": self.observation_pipeline.diagnostics(),
            "tau_velocity_scale_min": float(self._m7b_tau_velocity_scale.min().item()),
            "tau_velocity_scale_max": float(self._m7b_tau_velocity_scale.max().item()),
            "acceleration_limit_scale_min": float(self._m7b_acceleration_limit_scale.min().item()),
            "acceleration_limit_scale_max": float(self._m7b_acceleration_limit_scale.max().item()),
            "speed_limit_scale_min": float(self._m7b_speed_limit_scale.min().item()),
            "speed_limit_scale_max": float(self._m7b_speed_limit_scale.max().item()),
            "linear_drag_min": float(self._m7b_linear_drag.min().item()),
            "linear_drag_max": float(self._m7b_linear_drag.max().item()),
            "action_delay_steps_min": int(self._m7b_action_delay_steps.min().item()),
            "action_delay_steps_max": int(self._m7b_action_delay_steps.max().item()),
            "steady_wind_norm_max": float(torch.linalg.norm(self._wind_state.steady_wind_w, dim=1).max().item()),
            "gust_target_norm_max": float(torch.linalg.norm(self._wind_state.gust_target_w, dim=1).max().item()),
            "gust_current_norm_max": float(torch.linalg.norm(self._wind_state.gust_current_w, dim=1).max().item()),
            "current_wind_norm_max": float(torch.linalg.norm(wind, dim=1).max().item()),
            "tracking_saturation_fraction_max": float(
                self._saturation_fraction(self.acceleration_saturation_count, self._all_env_ids).max().item()
            ),
            "total_safety_saturation_fraction_max": float(
                self._saturation_fraction(self._m7b_total_safety_saturated_count, self._all_env_ids).max().item()
            ),
            "speed_saturation_fraction_max": float(self._saturation_fraction(self.speed_limit_count, self._all_env_ids).max().item()),
            "collision_risk_count_total": int(self.collision_risk_count.sum().item()),
            "workspace_violation_count_total": int(self.workspace_violation_count.sum().item()),
            "height_violation_count_total": int(self.height_violation_count.sum().item()),
        }

    def get_m7b_episode_history(self, clear: bool = False) -> list[dict[str, float | int | bool | str]]:
        return self.get_m6_episode_history(clear=clear)
