#!/usr/bin/env python3
"""Play a trained PPO checkpoint deterministically."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
import time
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic checkpoint playback for UAV rendezvous RL tasks.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-RL-v0", help="Gymnasium task ID.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for environment and PPO runner.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to latest run checkpoint.")
    parser.add_argument("--load_run", type=str, default=".*", help="Run directory regex when checkpoint is omitted.")
    parser.add_argument("--load_checkpoint", type=str, default="model_.*.pt", help="Checkpoint regex when omitted.")
    parser.add_argument("--steps", type=int, default=1000, help="Finite playback steps before exiting.")
    parser.add_argument("--real_time", action="store_true", default=False, help="Sleep to approximate real-time playback.")
    parser.add_argument(
        "--audit_hidden_state",
        action="store_true",
        default=False,
        help="Audit recurrent hidden-state reset behavior before playback.",
    )
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _assert_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite tensor detected in {name}.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(f"{name}.{key}", item)


def _resolve_checkpoint(agent_cfg: Any) -> str:
    if args_cli.checkpoint is not None:
        checkpoint = os.path.abspath(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.load_checkpoint)
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def _policy_device(policy_nn: Any) -> torch.device:
    return next(policy_nn.parameters()).device


def _obs_to_tensordict(policy_nn: Any, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
    device = _policy_device(policy_nn)
    batch_size = [int(obs_dict["policy"].shape[0])]
    return TensorDict({key: value.to(device=device) for key, value in obs_dict.items()}, batch_size=batch_size)


def _reset_policy(policy_nn: Any, dones: torch.Tensor | None = None) -> None:
    with torch.inference_mode():
        if dones is None:
            policy_nn.reset()
        else:
            policy_nn.reset(dones.to(device=_policy_device(policy_nn)))


def _deterministic_actions(policy_nn: Any, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    action_device = obs_dict["policy"].device
    actions = policy_nn.act_inference(_obs_to_tensordict(policy_nn, obs_dict))
    return actions.to(device=action_device)


def _clone_obs_dict(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in obs_dict.items()}


def _assert_recurrent_hidden_reset(policy_nn: Any, obs_dict: dict[str, torch.Tensor]) -> None:
    if not bool(getattr(policy_nn, "is_recurrent", False)):
        print("[INFO] Hidden-state audit skipped for feedforward policy.", flush=True)
        return
    if not isinstance(policy_nn.memory_a.rnn, torch.nn.GRU) or not isinstance(policy_nn.memory_c.rnn, torch.nn.GRU):
        raise RuntimeError("M6 hidden-state audit requires GRU actor and critic memories.")
    if policy_nn.memory_a is policy_nn.memory_c:
        raise RuntimeError("Actor and critic recurrent memories must be independent objects.")

    _reset_policy(policy_nn)
    obs_td = _obs_to_tensordict(policy_nn, obs_dict)
    with torch.inference_mode():
        policy_nn.act_inference(obs_td)
        policy_nn.evaluate(obs_td)
    actor_before = policy_nn.memory_a.hidden_state.detach().clone()
    critic_before = policy_nn.memory_c.hidden_state.detach().clone()
    if actor_before.shape[1] < 2:
        dones = torch.ones(actor_before.shape[1], dtype=torch.long, device=actor_before.device)
    else:
        dones = torch.zeros(actor_before.shape[1], dtype=torch.long, device=actor_before.device)
        dones[0] = 1
    _reset_policy(policy_nn, dones)
    actor_after = policy_nn.memory_a.hidden_state
    critic_after = policy_nn.memory_c.hidden_state
    done_mask = dones == 1
    keep_mask = ~done_mask
    if not bool(torch.all(actor_after[:, done_mask, :] == 0.0).item()):
        raise RuntimeError("Actor GRU hidden state was not cleared for done envs.")
    if not bool(torch.all(critic_after[:, done_mask, :] == 0.0).item()):
        raise RuntimeError("Critic GRU hidden state was not cleared for done envs.")
    if bool(torch.any(keep_mask).item()):
        torch.testing.assert_close(actor_after[:, keep_mask, :], actor_before[:, keep_mask, :])
        torch.testing.assert_close(critic_after[:, keep_mask, :], critic_before[:, keep_mask, :])
    _reset_policy(policy_nn)
    if policy_nn.memory_a.hidden_state is not None or policy_nn.memory_c.hidden_state is not None:
        raise RuntimeError("Full recurrent reset did not clear hidden states.")
    print("[INFO] Hidden-state reset audit passed.", flush=True)


def main() -> None:
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
    checkpoint = _resolve_checkpoint(agent_cfg)
    env_cfg.log_dir = os.path.dirname(checkpoint)

    gym_env = None
    rsl_env = None
    try:
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading model checkpoint from: {checkpoint}", flush=True)
        runner.load(checkpoint)
        policy_nn = runner.alg.policy
        policy_nn.to("cpu")
        policy_nn.eval()
        _reset_policy(policy_nn)
        obs_dict, _ = gym_env.reset()
        obs_dict = _clone_obs_dict(obs_dict)
        _assert_finite("reset_obs", obs_dict)
        if args_cli.audit_hidden_state:
            _assert_recurrent_hidden_reset(policy_nn, obs_dict)
        dt = gym_env.unwrapped.step_dt
        print(f"[INFO] Starting deterministic play loop for {args_cli.steps} steps.", flush=True)

        for step in range(args_cli.steps):
            start_time = time.time()
            with torch.inference_mode():
                actions = _deterministic_actions(policy_nn, obs_dict)
                if step == 0:
                    print(
                        f"[INFO] First inference action shape: {tuple(actions.shape)}, "
                        f"finite={bool(torch.isfinite(actions).all().item())}, "
                        f"abs_max={float(torch.abs(actions).max().item())}",
                        flush=True,
                    )
                obs_dict, rewards, terminated, truncated, _ = gym_env.step(actions)
                dones = (terminated | truncated).to(dtype=torch.long)
                _reset_policy(policy_nn, dones)
                obs_dict = _clone_obs_dict(obs_dict)
                if step == 0:
                    print(
                        f"[INFO] First deterministic env step completed: "
                        f"terminated={int(torch.count_nonzero(terminated).item())}, "
                        f"truncated={int(torch.count_nonzero(truncated).item())}, "
                        f"reward_mean={float(torch.mean(rewards).item())}",
                        flush=True,
                    )
            _assert_finite("obs", obs_dict)
            _assert_finite("rewards", rewards)
            if step == 0:
                print("[INFO] First deterministic post-step checks completed.", flush=True)
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        print(f"[INFO] play completed {args_cli.steps} steps for {gym_env.unwrapped.num_envs} environments.", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
