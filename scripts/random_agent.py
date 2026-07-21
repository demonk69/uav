#!/usr/bin/env python3
"""Run the M1 environment with random bounded actions for a finite number of steps."""

from __future__ import annotations

import argparse
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Random-action smoke agent for the UAV rendezvous M1 task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-Direct-v0", help="Gymnasium task ID.")
parser.add_argument("--steps", type=int, default=10000, help="Number of environment steps before exiting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uav_rendezvous_rl.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _assert_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite tensor detected in {name}.")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(f"{name}.{key}", item)


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        obs, _ = env.reset()
        _assert_finite("reset_obs", obs)

        for _ in range(args_cli.steps):
            actions = torch.empty(env.action_space.shape, device=env.unwrapped.device).uniform_(-1.0, 1.0)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            _assert_finite("obs", obs)
            _assert_finite("rewards", rewards)
            if terminated.shape != truncated.shape:
                raise RuntimeError("Terminated and truncated tensors have mismatched shapes.")

        print(f"[INFO] random_agent completed {args_cli.steps} steps for {env.unwrapped.num_envs} environments.", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
