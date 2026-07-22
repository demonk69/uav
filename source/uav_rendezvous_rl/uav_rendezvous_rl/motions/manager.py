"""Unified vectorized target motion manager for M3."""

from __future__ import annotations

import torch

from .base import MotionState
from .configs import (
    MODE_CONSTANT_ACCELERATION,
    MODE_CONSTANT_TURN,
    MODE_CONSTANT_VELOCITY,
    MODE_NAMES,
    MODE_PIECEWISE_ACCELERATION,
    TargetMotionManagerCfg,
    get_split_cfg,
    validate_motion_config,
)
from .constant_acceleration import compute_constant_acceleration
from .constant_turn import compute_constant_turn
from .constant_velocity import compute_constant_velocity
from .piecewise_acceleration import compute_piecewise_acceleration
from .sampling import randint_range, sample_acceleration, sample_mode_ids, uniform_range


class TargetMotionManager:
    """Batch target motion manager with mixed per-env modes."""

    def __init__(self, num_envs: int, cfg: TargetMotionManagerCfg, physics_dt: float, device: torch.device | str):
        validate_motion_config(cfg)
        self.num_envs = int(num_envs)
        self.cfg = cfg
        self.physics_dt = float(physics_dt)
        self.device = torch.device(device)
        self.p0_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.v0_w = torch.zeros_like(self.p0_w)
        self.mode_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.motion_step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.constant_acceleration_w = torch.zeros_like(self.p0_w)
        self.turn_omega = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.segment_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.segment_step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.segment_duration_steps = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.segment_start_position_w = torch.zeros_like(self.p0_w)
        self.segment_start_velocity_w = torch.zeros_like(self.p0_w)
        self.current_acceleration_w = torch.zeros_like(self.p0_w)
        self.segment_switch_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.invalid_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._split = "train"
        self._generator: torch.Generator | None = None

    @property
    def split(self) -> str:
        return self._split

    def reset(
        self,
        env_ids: torch.Tensor,
        initial_position_w: torch.Tensor,
        initial_velocity_w: torch.Tensor,
        generator: torch.Generator,
        split: str,
    ) -> MotionState:
        """Reset selected environments with independent modes and parameters."""

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        split_cfg = get_split_cfg(self.cfg, split)
        self._split = split
        self._generator = generator
        count = int(env_ids.numel())
        self.p0_w[env_ids] = initial_position_w.to(device=self.device, dtype=torch.float32)
        self.v0_w[env_ids] = initial_velocity_w.to(device=self.device, dtype=torch.float32)
        self.v0_w[env_ids, 2] = 0.0
        self.mode_id[env_ids] = sample_mode_ids(
            count, env_ids, split_cfg, generator, self.device, self.cfg.force_mode_cycle_on_reset
        )
        self.motion_step_count[env_ids] = 0
        self.constant_acceleration_w[env_ids] = sample_acceleration(
            count, split_cfg.acceleration_x_range, split_cfg.acceleration_y_range, generator, self.device
        )
        self.turn_omega[env_ids] = uniform_range(split_cfg.turn_omega_range, (count,), generator, self.device)
        self.segment_index[env_ids] = 0
        self.segment_step_count[env_ids] = 0
        self.segment_duration_steps[env_ids] = randint_range(
            split_cfg.piecewise_segment_duration_steps_range, (count,), generator, self.device
        )
        self.segment_start_position_w[env_ids] = self.p0_w[env_ids]
        self.segment_start_velocity_w[env_ids] = self.v0_w[env_ids]
        self.current_acceleration_w[env_ids] = sample_acceleration(
            count,
            split_cfg.piecewise_acceleration_x_range,
            split_cfg.piecewise_acceleration_y_range,
            generator,
            self.device,
        )
        self.segment_switch_count[env_ids] = 0
        state = self.evaluate()
        self.invalid_mask[env_ids] = self._validate_state(state)[env_ids]
        return state

    def step(self) -> MotionState:
        """Advance one physics step and evaluate all environments."""

        if self._generator is None:
            raise RuntimeError("TargetMotionManager.step() called before reset().")
        self.motion_step_count += 1
        piecewise_mask = self.mode_id == MODE_PIECEWISE_ACCELERATION
        self.segment_step_count[piecewise_mask] += 1
        switch_mask = piecewise_mask & (self.segment_step_count >= self.segment_duration_steps)
        if bool(torch.any(switch_mask).item()):
            self._switch_piecewise_segments(switch_mask)
        state = self.evaluate()
        self.invalid_mask[:] = self._validate_state(state)
        return state

    def evaluate(self, physics_step_count: torch.Tensor | None = None) -> MotionState:
        """Evaluate all modes without mutating segment state."""

        if physics_step_count is None:
            step_count = self.motion_step_count
        else:
            step_count = torch.as_tensor(physics_step_count, dtype=torch.long, device=self.device)
            if step_count.ndim == 0:
                step_count = step_count.repeat(self.num_envs)
        self._validate_mode_ids()
        cv = compute_constant_velocity(self.p0_w, self.v0_w, step_count, self.physics_dt, self.cfg.fixed_height)
        ca = compute_constant_acceleration(
            self.p0_w, self.v0_w, self.constant_acceleration_w, step_count, self.physics_dt, self.cfg.fixed_height
        )
        ct = compute_constant_turn(
            self.p0_w,
            self.v0_w,
            self.turn_omega,
            step_count,
            self.physics_dt,
            self.cfg.fixed_height,
            self.cfg.omega_epsilon,
        )
        pw = compute_piecewise_acceleration(
            self.segment_start_position_w,
            self.segment_start_velocity_w,
            self.current_acceleration_w,
            self.segment_step_count,
            self.physics_dt,
            self.cfg.fixed_height,
        )
        p_w = torch.empty_like(self.p0_w)
        v_w = torch.empty_like(self.v0_w)
        a_w = torch.empty_like(self.v0_w)
        self._select_mode_state(p_w, v_w, a_w, cv, MODE_CONSTANT_VELOCITY)
        self._select_mode_state(p_w, v_w, a_w, ca, MODE_CONSTANT_ACCELERATION)
        self._select_mode_state(p_w, v_w, a_w, ct, MODE_CONSTANT_TURN)
        self._select_mode_state(p_w, v_w, a_w, pw, MODE_PIECEWISE_ACCELERATION)
        return MotionState(p_w, v_w, a_w)

    def _select_mode_state(
        self, p_w: torch.Tensor, v_w: torch.Tensor, a_w: torch.Tensor, state: MotionState, mode_id: int
    ) -> None:
        mask = self.mode_id == mode_id
        if bool(torch.any(mask).item()):
            p_w[mask] = state.p_target_w[mask]
            v_w[mask] = state.v_target_w[mask]
            a_w[mask] = state.a_target_w[mask]

    def _switch_piecewise_segments(self, switch_mask: torch.Tensor) -> None:
        boundary_state = compute_piecewise_acceleration(
            self.segment_start_position_w,
            self.segment_start_velocity_w,
            self.current_acceleration_w,
            self.segment_duration_steps,
            self.physics_dt,
            self.cfg.fixed_height,
        )
        split_cfg = get_split_cfg(self.cfg, self._split)
        count = int(torch.count_nonzero(switch_mask).item())
        self.segment_start_position_w[switch_mask] = boundary_state.p_target_w[switch_mask]
        self.segment_start_velocity_w[switch_mask] = boundary_state.v_target_w[switch_mask]
        self.segment_step_count[switch_mask] = 0
        self.segment_index[switch_mask] += 1
        self.segment_switch_count[switch_mask] += 1
        self.segment_duration_steps[switch_mask] = randint_range(
            split_cfg.piecewise_segment_duration_steps_range, (count,), self._generator, self.device
        )
        self.current_acceleration_w[switch_mask] = sample_acceleration(
            count,
            split_cfg.piecewise_acceleration_x_range,
            split_cfg.piecewise_acceleration_y_range,
            self._generator,
            self.device,
        )

    def _validate_mode_ids(self) -> None:
        legal = (self.mode_id >= MODE_CONSTANT_VELOCITY) & (self.mode_id <= MODE_PIECEWISE_ACCELERATION)
        if not bool(torch.all(legal).item()):
            invalid = self.mode_id[~legal].detach().cpu().tolist()
            raise RuntimeError(f"Unknown target motion mode_id values: {invalid}.")

    def _validate_state(self, state: MotionState) -> torch.Tensor:
        tensors_finite = torch.isfinite(state.p_target_w).all(dim=1)
        tensors_finite &= torch.isfinite(state.v_target_w).all(dim=1)
        tensors_finite &= torch.isfinite(state.a_target_w).all(dim=1)
        speed_valid = torch.linalg.norm(state.v_target_w, dim=1) <= self.cfg.max_speed
        accel_valid = torch.linalg.norm(state.a_target_w, dim=1) <= self.cfg.max_acceleration
        height_valid = (state.p_target_w[:, 2] >= self.cfg.workspace_z_range[0]) & (
            state.p_target_w[:, 2] <= self.cfg.workspace_z_range[1]
        )
        workspace_valid = (state.p_target_w[:, 0] >= self.cfg.workspace_x_range[0]) & (
            state.p_target_w[:, 0] <= self.cfg.workspace_x_range[1]
        )
        workspace_valid &= (state.p_target_w[:, 1] >= self.cfg.workspace_y_range[0]) & (
            state.p_target_w[:, 1] <= self.cfg.workspace_y_range[1]
        )
        duration_valid = self.segment_duration_steps >= 1
        mode_valid = (self.mode_id >= MODE_CONSTANT_VELOCITY) & (self.mode_id <= MODE_PIECEWISE_ACCELERATION)
        valid = tensors_finite & speed_valid & accel_valid & height_valid & workspace_valid & duration_valid & mode_valid
        return ~valid

    def mode_counts(self) -> dict[str, int]:
        """Return current mode counts using CPU scalars."""

        return {name: int(torch.count_nonzero(self.mode_id == mode_id).item()) for mode_id, name in MODE_NAMES.items()}

    def max_speed(self, state: MotionState) -> float:
        return float(torch.linalg.norm(state.v_target_w, dim=1).max().item())

    def max_acceleration(self, state: MotionState) -> float:
        return float(torch.linalg.norm(state.a_target_w, dim=1).max().item())
