#!/usr/bin/env python3
"""Train the M5 feedforward PPO agent with RSL-RL."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train feedforward PPO for the M5 UAV rendezvous RL task.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-RL-v0", help="Gymnasium task ID.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for environment and PPO runner.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Number of PPO learning iterations.")
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
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402


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
