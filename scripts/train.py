#!/usr/bin/env python3
"""Train PPO agents for the UAV rendezvous RL tasks with RSL-RL."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from datetime import datetime
from types import MethodType

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO for the UAV rendezvous RL tasks.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-RL-v0", help="Gymnasium task ID.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for environment and PPO runner.")
    parser.add_argument("--num_steps_per_env", type=int, default=None, help="PPO rollout length per environment.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Number of PPO learning iterations.")
    parser.add_argument(
        "--target_motion_mode",
        choices=("Mixed", "ConstantVelocity", "ConstantAcceleration", "ConstantTurn", "PiecewiseAcceleration"),
        default=None,
        help="Override target-motion mode distribution for controlled M6 training.",
    )
    parser.add_argument(
        "--force_mode_cycle_on_reset",
        action="store_true",
        default=False,
        help="Cycle modes by env id on reset. Intended for balanced mixed-mode evaluation/audits.",
    )
    parser.add_argument(
        "--m7a_stage",
        type=str,
        default=None,
        help="Override M7A observation-degradation stage: 0, 1, 2, 3, or 4.",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional run-name suffix.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from a previous run.")
    parser.add_argument("--load_run", type=str, default=None, help="Run directory regex for resume.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint regex for resume.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from uav_rendezvous_rl.observations import make_m7a_observation_cfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mode_probabilities(mode_name: str) -> tuple[float, float, float, float]:
    if mode_name == "Mixed":
        return (0.25, 0.25, 0.25, 0.25)
    if mode_name == "ConstantVelocity":
        return (1.0, 0.0, 0.0, 0.0)
    if mode_name == "ConstantAcceleration":
        return (0.0, 1.0, 0.0, 0.0)
    if mode_name == "ConstantTurn":
        return (0.0, 0.0, 1.0, 0.0)
    if mode_name == "PiecewiseAcceleration":
        return (0.0, 0.0, 0.0, 1.0)
    raise RuntimeError(f"Unknown target motion mode: {mode_name}")


def _configure_target_motion(env_cfg: object, mode_name: str | None, force_cycle: bool) -> None:
    if mode_name is None and not force_cycle:
        return
    probabilities = _mode_probabilities(mode_name or "Mixed")
    motion_cfg = env_cfg.target_motion
    env_cfg.target_motion = replace(
        motion_cfg,
        force_mode_cycle_on_reset=force_cycle,
        train=replace(motion_cfg.train, mode_probabilities=probabilities),
        validation=replace(motion_cfg.validation, mode_probabilities=probabilities),
        test=replace(motion_cfg.test, mode_probabilities=probabilities),
    )


def _configure_m7a_observation(env_cfg: object, stage: str | None) -> None:
    if stage is None:
        return
    if not hasattr(env_cfg, "observation_degradation"):
        raise RuntimeError("--m7a_stage can only be used with M7A tasks.")
    env_cfg.observation_degradation = make_m7a_observation_cfg(stage)


def _hidden_norm(hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None) -> float:
    if hidden_state is None:
        return 0.0
    if isinstance(hidden_state, tuple):
        norms = [torch.linalg.vector_norm(item.detach()).item() for item in hidden_state]
        return float(sum(norms))
    return float(torch.linalg.vector_norm(hidden_state.detach()).item())


def _install_recurrent_hidden_norm_logging(runner: OnPolicyRunner) -> None:
    policy = runner.alg.policy
    if not bool(getattr(policy, "is_recurrent", False)):
        return
    original_log = runner.log

    def log_with_hidden_norms(self: OnPolicyRunner, locs: dict, *args, **kwargs) -> None:
        original_log(locs, *args, **kwargs)
        actor_hidden_norm = _hidden_norm(policy.memory_a.hidden_state)
        critic_hidden_norm = _hidden_norm(policy.memory_c.hidden_state)
        if self.writer is not None:
            self.writer.add_scalar("Policy/actor_hidden_norm", actor_hidden_norm, locs["it"])
            self.writer.add_scalar("Policy/critic_hidden_norm", critic_hidden_norm, locs["it"])
        print(
            f"[INFO] Hidden norms at iteration {locs['it']}: "
            f"actor={actor_hidden_norm:.6f}, critic={critic_hidden_norm:.6f}",
            flush=True,
        )

    runner.log = MethodType(log_with_hidden_norms, runner)


def _assert_recurrent_policy_contract(runner: OnPolicyRunner, task_id: str) -> None:
    policy = runner.alg.policy
    is_recurrent = bool(getattr(policy, "is_recurrent", False))
    recurrent_task_ids = {"Isaac-Uav-Rendezvous-Recurrent-v0", "Isaac-Uav-Rendezvous-M7A-GRU-v0"}
    if task_id not in recurrent_task_ids and not is_recurrent:
        return

    policy_class = policy.__class__.__name__
    _require(policy_class == "ActorCriticRecurrent", f"Expected ActorCriticRecurrent, got {policy_class}.")
    _require(is_recurrent, "Recurrent policy did not report is_recurrent=True.")
    _require(hasattr(policy, "memory_a") and hasattr(policy, "memory_c"), "Recurrent policy missing actor/critic memory.")
    _require(policy.memory_a is not policy.memory_c, "Actor and critic recurrent memories must be independent objects.")
    _require(isinstance(policy.memory_a.rnn, torch.nn.GRU), "Actor memory is not a GRU.")
    _require(isinstance(policy.memory_c.rnn, torch.nn.GRU), "Critic memory is not a GRU.")

    actor_input_dim = int(policy.memory_a.rnn.input_size)
    critic_input_dim = int(policy.memory_c.rnn.input_size)
    action_dim = int(runner.env.num_actions)
    _require(actor_input_dim == 25, f"Expected actor input dim 25, got {actor_input_dim}.")
    _require(critic_input_dim == 57, f"Expected critic input dim 57, got {critic_input_dim}.")
    _require(action_dim == 3, f"Expected action dim 3, got {action_dim}.")

    print(f"[INFO] policy class: {policy_class}", flush=True)
    print(f"[INFO] is_recurrent={is_recurrent}", flush=True)
    print(f"[INFO] actor memory: {policy.memory_a.rnn.__class__.__name__}", flush=True)
    print(f"[INFO] critic memory: {policy.memory_c.rnn.__class__.__name__}", flush=True)
    print(f"[INFO] policy input dim: {actor_input_dim}", flush=True)
    print(f"[INFO] critic input dim: {critic_input_dim}", flush=True)
    print(f"[INFO] action dim: {action_dim}", flush=True)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed
    agent_cfg.device = device
    _configure_target_motion(env_cfg, args_cli.target_motion_mode, args_cli.force_mode_cycle_on_reset)
    _configure_m7a_observation(env_cfg, args_cli.m7a_stage)
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir_name}", flush=True)
    if agent_cfg.run_name:
        log_dir_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir_name)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        _assert_recurrent_policy_contract(runner, args_cli.task)
        _install_recurrent_hidden_norm_logging(runner)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
            runner.load(resume_path)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        start_time = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"[INFO] Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
