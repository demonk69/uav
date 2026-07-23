#!/usr/bin/env python3
"""Audit that the M6 GRU policy path can condition actions on observation history."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M6 GRU history-sensitivity audit.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=4, help="Batch size for synthetic observation histories.")
    parser.add_argument("--history_steps", type=int, default=8, help="Number of distinct history observations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for environment and PPO runner.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-Recurrent-v0", help="Gymnasium M6 task ID.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional trained recurrent checkpoint.")
    parser.add_argument(
        "--feedforward_task",
        type=str,
        default="Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0",
        help="Gymnasium task ID for the fair feedforward ablation.",
    )
    parser.add_argument("--feedforward_checkpoint", type=str, default=None, help="Optional trained feedforward checkpoint.")
    parser.add_argument("--oracle_gain", type=float, default=0.8, help="Current-state proportional oracle gain.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from uav_rendezvous_rl.controllers import clamp_vector_norm  # noqa: E402
from uav_rendezvous_rl.mdp import raw_action_from_velocity_command  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _make_obs(policy_obs: torch.Tensor, critic_obs: torch.Tensor) -> TensorDict:
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[int(policy_obs.shape[0])])


def _obs_from_policy(policy_obs: torch.Tensor) -> TensorDict:
    critic_obs = torch.zeros((policy_obs.shape[0], 57), dtype=torch.float32, device=policy_obs.device)
    critic_obs[:, 0:25] = policy_obs
    return _make_obs(policy_obs, critic_obs)


def _base_policy_obs(num_envs: int, device: torch.device | str) -> torch.Tensor:
    obs = torch.zeros((num_envs, 25), dtype=torch.float32, device=device)
    obs[:, 0] = -5.0
    obs[:, 9] = 1.0
    obs[:, 13] = 1.0
    obs[:, 21] = 5.0
    obs[:, 24] = 5.0
    return obs


def _synthetic_histories(
    num_envs: int, steps: int, device: torch.device | str
) -> list[tuple[str, list[TensorDict], list[TensorDict], TensorDict, TensorDict]]:
    final_policy_a = _base_policy_obs(num_envs, device)
    final_policy_b = final_policy_a.clone()
    final_obs_a = _obs_from_policy(final_policy_a)
    final_obs_b = _obs_from_policy(final_policy_b)

    histories = []

    accel_a = []
    accel_b = []
    history_a = []
    history_b = []
    for step in range(steps):
        obs_a = _base_policy_obs(num_envs, device)
        obs_b = _base_policy_obs(num_envs, device)
        obs_a[:, 0] = -5.0 + 0.05 * step
        obs_b[:, 0] = -5.0 - 0.05 * step
        obs_a[:, 3] = 0.10 * step
        obs_b[:, 3] = -0.10 * step
        accel_a.append(_obs_from_policy(obs_a))
        accel_b.append(_obs_from_policy(obs_b))
    histories.append(("accelerating_vs_decelerating", accel_a, accel_b, final_obs_a, final_obs_b))

    turn_a = []
    turn_b = []
    for step in range(steps):
        angle = 0.12 * step
        obs_a = _base_policy_obs(num_envs, device)
        obs_b = _base_policy_obs(num_envs, device)
        obs_a[:, 0] = -5.0 * torch.cos(torch.tensor(angle, device=device))
        obs_a[:, 1] = 5.0 * torch.sin(torch.tensor(angle, device=device))
        obs_b[:, 0] = -5.0 * torch.cos(torch.tensor(angle, device=device))
        obs_b[:, 1] = -5.0 * torch.sin(torch.tensor(angle, device=device))
        obs_a[:, 3] = 0.4 * torch.sin(torch.tensor(angle, device=device))
        obs_a[:, 4] = 0.4 * torch.cos(torch.tensor(angle, device=device))
        obs_b[:, 3] = -0.4 * torch.sin(torch.tensor(angle, device=device))
        obs_b[:, 4] = 0.4 * torch.cos(torch.tensor(angle, device=device))
        turn_a.append(_obs_from_policy(obs_a))
        turn_b.append(_obs_from_policy(obs_b))
    histories.append(("positive_turn_vs_negative_turn", turn_a, turn_b, final_obs_a, final_obs_b))

    pwa_a = []
    pwa_b = []
    midpoint = max(1, steps // 2)
    for step in range(steps):
        obs_a = _base_policy_obs(num_envs, device)
        obs_b = _base_policy_obs(num_envs, device)
        sign_a = 1.0 if step < midpoint else -1.0
        sign_b = -1.0 if step < midpoint else 1.0
        obs_a[:, 3] = sign_a * (0.15 + 0.03 * step)
        obs_b[:, 3] = sign_b * (0.15 + 0.03 * step)
        obs_a[:, 4] = -sign_a * 0.10
        obs_b[:, 4] = -sign_b * 0.10
        pwa_a.append(_obs_from_policy(obs_a))
        pwa_b.append(_obs_from_policy(obs_b))
    histories.append(("pwa_previous_segment_difference", pwa_a, pwa_b, final_obs_a, final_obs_b))

    return histories


def _run_recurrent_sequence(
    policy: torch.nn.Module, history: list[TensorDict], final_obs: TensorDict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        policy.reset()
        for obs in history:
            policy.act_inference(obs)
            policy.evaluate(obs)
        actor_hidden_before_final = policy.memory_a.hidden_state.detach().clone()
        critic_hidden_before_final = policy.memory_c.hidden_state.detach().clone()
        final_actions = policy.act_inference(final_obs).detach().clone()
    return actor_hidden_before_final, critic_hidden_before_final, final_actions


def _feedforward_action(policy: torch.nn.Module, final_obs: TensorDict) -> torch.Tensor:
    with torch.inference_mode():
        return policy.act_inference(final_obs).detach().clone()


def _oracle_actions(final_obs: TensorDict, v_max: float, oracle_gain: float) -> torch.Tensor:
    policy_obs = final_obs["policy"]
    p_rel_w = policy_obs[:, 0:3]
    v_rel_w = policy_obs[:, 3:6]
    v_ego_w = policy_obs[:, 6:9]
    b_des_w = policy_obs[:, 21:24]
    v_target_w = v_rel_w + v_ego_w
    v_des_w = v_target_w + float(oracle_gain) * (p_rel_w + b_des_w)
    v_cmd_w, _ = clamp_vector_norm(v_des_w, float(v_max) * 0.98)
    return raw_action_from_velocity_command(v_cmd_w, float(v_max))


def _l2_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((left - right).detach()).item())


def _load_policy(task_id: str, checkpoint: str | None, device: str) -> tuple[Any, Any, Any]:
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    gym_env = gym.make(task_id, cfg=env_cfg)
    rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if checkpoint is not None:
        checkpoint = os.path.abspath(checkpoint)
        _assert(os.path.isfile(checkpoint), f"Checkpoint does not exist: {checkpoint}")
        runner.load(checkpoint)
    runner.alg.policy.eval()
    return gym_env, rsl_env, runner


def _load_policy_on_env(task_id: str, checkpoint: str | None, rsl_env: RslRlVecEnvWrapper, device: str) -> Any:
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if checkpoint is not None:
        checkpoint = os.path.abspath(checkpoint)
        _assert(os.path.isfile(checkpoint), f"Checkpoint does not exist: {checkpoint}")
        runner.load(checkpoint)
    runner.alg.policy.eval()
    return runner


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    gym_env = None
    rsl_env = None
    try:
        gym_env, rsl_env, runner = _load_policy(args_cli.task, args_cli.checkpoint, device)
        policy = runner.alg.policy
        _assert(bool(getattr(policy, "is_recurrent", False)), "History sensitivity requires a recurrent policy.")
        _assert(isinstance(policy.memory_a.rnn, torch.nn.GRU), "Actor memory must be GRU.")

        ff_policy = None
        if args_cli.feedforward_checkpoint is not None:
            ff_runner = _load_policy_on_env(
                args_cli.feedforward_task, args_cli.feedforward_checkpoint, rsl_env, device
            )
            ff_policy = ff_runner.alg.policy
            _assert(not bool(getattr(ff_policy, "is_recurrent", False)), "Feedforward checkpoint loaded recurrent policy.")

        histories = _synthetic_histories(args_cli.num_envs, args_cli.history_steps, device)
        pair_reports = {}
        for pair_name, history_a, history_b, final_obs_a, final_obs_b in histories:
            final_obs_delta = float(torch.max(torch.abs(final_obs_a["policy"] - final_obs_b["policy"])).item())
            _assert(final_obs_delta <= 1.0e-7, f"Final observations differ for {pair_name}: {final_obs_delta}")
            actor_hidden_a, critic_hidden_a, gru_actions_a = _run_recurrent_sequence(policy, history_a, final_obs_a)
            actor_hidden_b, critic_hidden_b, gru_actions_b = _run_recurrent_sequence(policy, history_b, final_obs_b)
            policy.reset()

            feedforward_action_distance = None
            feedforward_oracle_error = None
            if ff_policy is not None:
                ff_actions_a = _feedforward_action(ff_policy, final_obs_a)
                ff_actions_b = _feedforward_action(ff_policy, final_obs_b)
                feedforward_action_distance = _l2_distance(ff_actions_a, ff_actions_b)
                oracle = _oracle_actions(final_obs_a, gym_env.unwrapped.cfg.action.v_max, args_cli.oracle_gain)
                feedforward_oracle_error = 0.5 * (
                    _l2_distance(ff_actions_a, oracle) + _l2_distance(ff_actions_b, oracle)
                )
            else:
                oracle = _oracle_actions(final_obs_a, gym_env.unwrapped.cfg.action.v_max, args_cli.oracle_gain)

            actor_hidden_distance = _l2_distance(actor_hidden_a, actor_hidden_b)
            critic_hidden_distance = _l2_distance(critic_hidden_a, critic_hidden_b)
            gru_action_distance = _l2_distance(gru_actions_a, gru_actions_b)
            gru_oracle_error = 0.5 * (_l2_distance(gru_actions_a, oracle) + _l2_distance(gru_actions_b, oracle))
            _assert(actor_hidden_distance > 1.0e-6, f"{pair_name} did not change actor hidden state.")

            pair_reports[pair_name] = {
                "final_observation_max_abs_diff": final_obs_delta,
                "actor_hidden_distance": actor_hidden_distance,
                "critic_hidden_distance": critic_hidden_distance,
                "gru_action_distance": gru_action_distance,
                "feedforward_action_distance": feedforward_action_distance,
                "gru_oracle_action_error": gru_oracle_error,
                "feedforward_oracle_action_error": feedforward_oracle_error,
            }

        report = {
            "task": args_cli.task,
            "checkpoint": os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint is not None else None,
            "feedforward_task": args_cli.feedforward_task if args_cli.feedforward_checkpoint is not None else None,
            "feedforward_checkpoint": os.path.abspath(args_cli.feedforward_checkpoint)
            if args_cli.feedforward_checkpoint is not None
            else None,
            "num_envs": int(args_cli.num_envs),
            "history_steps": int(args_cli.history_steps),
            "history_pairs": pair_reports,
        }
        print(f"[INFO] M6 history sensitivity audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
