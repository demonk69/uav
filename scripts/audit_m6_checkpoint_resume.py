#!/usr/bin/env python3
"""Formal checkpoint resume audit for M6 mixed-mode GRU PPO."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M6 recurrent checkpoint resume audit.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for environment and PPO runner.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-Recurrent-v0", help="Gymnasium M6 task ID.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Mixed-mode GRU checkpoint to resume from.")
    parser.add_argument("--resume_iterations", type=int, default=3, help="Number of iterations to continue training.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _save_runner_checkpoint(runner: OnPolicyRunner, path: str) -> None:
    torch.save(
        {
            "model_state_dict": _to_cpu(runner.alg.policy.state_dict()),
            "optimizer_state_dict": _to_cpu(runner.alg.optimizer.state_dict()),
            "iter": runner.current_learning_iteration,
            "infos": {"audit": "m6_checkpoint_resume"},
        },
        path,
    )


def _flat_parameters(policy: torch.nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().cpu().reshape(-1) for param in policy.parameters()])


def _hidden_norm(hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None) -> float:
    if hidden_state is None:
        return 0.0
    if isinstance(hidden_state, tuple):
        return float(sum(torch.linalg.vector_norm(item.detach()).item() for item in hidden_state))
    return float(torch.linalg.vector_norm(hidden_state.detach()).item())


def _install_finite_monitors(runner: OnPolicyRunner, rsl_env: RslRlVecEnvWrapper) -> dict[str, Any]:
    summary: dict[str, Any] = {"losses": [], "reward_sum": 0.0, "reward_steps": 0, "done_count": 0}
    original_step = rsl_env.step
    original_update = runner.alg.update

    def monitored_step(actions: torch.Tensor):
        obs, rewards, dones, extras = original_step(actions)
        _assert(bool(torch.isfinite(rewards).all().item()), "Non-finite reward during resume audit.")
        summary["reward_sum"] += float(torch.sum(rewards).item())
        summary["reward_steps"] += int(rewards.numel())
        summary["done_count"] += int(torch.count_nonzero(dones).item())
        return obs, rewards, dones, extras

    def monitored_update():
        loss_dict = original_update()
        for key, value in loss_dict.items():
            _assert(math.isfinite(float(value)), f"Non-finite {key} loss during resume audit.")
        summary["losses"].append({key: float(value) for key, value in loss_dict.items()})
        return loss_dict

    rsl_env.step = monitored_step
    runner.alg.update = monitored_update
    return summary


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    checkpoint = os.path.abspath(args_cli.checkpoint)
    _assert(os.path.isfile(checkpoint), f"Checkpoint does not exist: {checkpoint}")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device

    gym_env = None
    rsl_env = None
    try:
        print("[INFO] Creating M6 resume-audit environment.", flush=True)
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        print("[INFO] Wrapping resume-audit environment for RSL-RL.", flush=True)
        rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        print("[INFO] Loading source checkpoint for resume audit.", flush=True)
        source_runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        source_runner.load(checkpoint)
        save_iteration = int(source_runner.current_learning_iteration)

        temp_parent = "/tmp/opencode" if os.path.isdir("/tmp/opencode") else None
        with tempfile.TemporaryDirectory(prefix="m6_resume_", dir=temp_parent) as temp_dir:
            saved_checkpoint = os.path.join(temp_dir, "resume_source.pt")
            print(f"[INFO] Saving normalized resume source checkpoint to {saved_checkpoint}.", flush=True)
            _save_runner_checkpoint(source_runner, saved_checkpoint)
            print("[INFO] Loading normalized resume source checkpoint.", flush=True)
            runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=temp_dir, device=agent_cfg.device)
            infos = runner.load(saved_checkpoint, map_location=agent_cfg.device)
            loaded_iteration = int(runner.current_learning_iteration)
            optimizer_state_exists = len(runner.alg.optimizer.state_dict().get("state", {})) > 0
            actor_hidden_initial_norm = _hidden_norm(runner.alg.policy.memory_a.hidden_state)
            critic_hidden_initial_norm = _hidden_norm(runner.alg.policy.memory_c.hidden_state)
            _assert(actor_hidden_initial_norm == 0.0, "Actor hidden state was not empty before resumed rollout.")
            _assert(critic_hidden_initial_norm == 0.0, "Critic hidden state was not empty before resumed rollout.")

            before_params = _flat_parameters(runner.alg.policy)
            monitor = _install_finite_monitors(runner, rsl_env)
            print("[INFO] Running resumed learning iterations.", flush=True)
            runner.learn(num_learning_iterations=int(args_cli.resume_iterations), init_at_random_ep_len=True)
            print("[INFO] Resumed learning iterations completed.", flush=True)
            after_params = _flat_parameters(runner.alg.policy)
            parameter_change_norm = float(torch.linalg.vector_norm(after_params - before_params).item())
            resumed_final_iteration = int(runner.current_learning_iteration)
            _assert(parameter_change_norm > 0.0, "Resumed training did not change policy parameters.")
            _assert(len(monitor["losses"]) >= int(args_cli.resume_iterations), "Not all resumed iterations produced losses.")

        report = {
            "task": args_cli.task,
            "checkpoint": checkpoint,
            "save_iteration": save_iteration,
            "loaded_iteration": loaded_iteration,
            "resumed_final_iteration": resumed_final_iteration,
            "resume_iterations": int(args_cli.resume_iterations),
            "optimizer_state_exists": optimizer_state_exists,
            "infos": infos,
            "parameter_change_norm": parameter_change_norm,
            "actor_hidden_initial_norm": actor_hidden_initial_norm,
            "critic_hidden_initial_norm": critic_hidden_initial_norm,
            "reward_sum": float(monitor["reward_sum"]),
            "reward_steps": int(monitor["reward_steps"]),
            "done_count": int(monitor["done_count"]),
            "losses": monitor["losses"],
        }
        print(f"[INFO] M6 checkpoint resume audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
