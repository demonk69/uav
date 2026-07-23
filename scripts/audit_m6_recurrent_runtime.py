#!/usr/bin/env python3
"""Runtime audit for the M6 recurrent PPO plumbing under Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime audit for M6 recurrent PPO wiring.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
    parser.add_argument("--steps", type=int, default=64, help="Short recurrent rollout steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for environment and PPO runner.")
    parser.add_argument("--task", type=str, default="Isaac-Uav-Rendezvous-Recurrent-v0", help="Gymnasium M6 task ID.")
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
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _assert(bool(torch.isfinite(value).all().item()), f"Non-finite tensor detected in {name}.")
    elif isinstance(value, TensorDict):
        for key, item in value.items():
            _assert_finite(f"{name}.{key}", item)


def _assert_policy_contract(runner: OnPolicyRunner) -> dict[str, Any]:
    policy = runner.alg.policy
    policy_class = policy.__class__.__name__
    _assert(policy_class == "ActorCriticRecurrent", f"Expected ActorCriticRecurrent, got {policy_class}.")
    _assert(bool(getattr(policy, "is_recurrent", False)), "Policy did not report is_recurrent=True.")
    _assert(hasattr(policy, "memory_a") and hasattr(policy, "memory_c"), "Policy missing recurrent memories.")
    _assert(policy.memory_a is not policy.memory_c, "Actor and critic memories must be independent objects.")
    _assert(isinstance(policy.memory_a.rnn, torch.nn.GRU), "Actor memory is not GRU.")
    _assert(isinstance(policy.memory_c.rnn, torch.nn.GRU), "Critic memory is not GRU.")
    actor_input_dim = int(policy.memory_a.rnn.input_size)
    critic_input_dim = int(policy.memory_c.rnn.input_size)
    action_dim = int(runner.env.num_actions)
    _assert(actor_input_dim == 25, f"Expected actor input dim 25, got {actor_input_dim}.")
    _assert(critic_input_dim == 57, f"Expected critic input dim 57, got {critic_input_dim}.")
    _assert(action_dim == 3, f"Expected action dim 3, got {action_dim}.")
    return {
        "policy_class": policy_class,
        "is_recurrent": True,
        "actor_memory": policy.memory_a.rnn.__class__.__name__,
        "critic_memory": policy.memory_c.rnn.__class__.__name__,
        "actor_input_dim": actor_input_dim,
        "critic_input_dim": critic_input_dim,
        "action_dim": action_dim,
    }


def _audit_hidden_reset(runner: OnPolicyRunner, obs: TensorDict) -> dict[str, Any]:
    policy = runner.alg.policy
    policy.reset()
    print("[INFO] Populating actor and critic GRU hidden states.", flush=True)
    with torch.inference_mode():
        policy.act_inference(obs)
        policy.evaluate(obs)
    actor_before = policy.memory_a.hidden_state.detach().clone()
    critic_before = policy.memory_c.hidden_state.detach().clone()
    print("[INFO] Applying partial done mask to GRU hidden states.", flush=True)
    if actor_before.shape[1] < 2:
        dones = torch.ones(actor_before.shape[1], dtype=torch.long, device=actor_before.device)
    else:
        dones = torch.zeros(actor_before.shape[1], dtype=torch.long, device=actor_before.device)
        dones[0] = 1
    with torch.inference_mode():
        policy.reset(dones)
    actor_after = policy.memory_a.hidden_state
    critic_after = policy.memory_c.hidden_state
    done_mask = dones == 1
    keep_mask = ~done_mask
    _assert(bool(torch.all(actor_after[:, done_mask, :] == 0.0).item()), "Actor done hidden state not zeroed.")
    _assert(bool(torch.all(critic_after[:, done_mask, :] == 0.0).item()), "Critic done hidden state not zeroed.")
    if bool(torch.any(keep_mask).item()):
        actor_keep_delta = float(torch.max(torch.abs(actor_after[:, keep_mask, :] - actor_before[:, keep_mask, :])).item())
        critic_keep_delta = float(torch.max(torch.abs(critic_after[:, keep_mask, :] - critic_before[:, keep_mask, :])).item())
        _assert(actor_keep_delta <= 1.0e-6, f"Actor non-done hidden state changed: {actor_keep_delta}.")
        _assert(critic_keep_delta <= 1.0e-6, f"Critic non-done hidden state changed: {critic_keep_delta}.")
    print("[INFO] Applying full GRU hidden-state reset.", flush=True)
    policy.reset()
    _assert(policy.memory_a.hidden_state is None, "Full actor hidden reset failed.")
    _assert(policy.memory_c.hidden_state is None, "Full critic hidden reset failed.")
    return {"done_count": int(torch.count_nonzero(dones).item()), "kept_count": int(torch.count_nonzero(keep_mask).item())}


def _audit_checkpoint(runner: OnPolicyRunner, rsl_env: RslRlVecEnvWrapper, agent_cfg: Any) -> dict[str, Any]:
    def to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(to_cpu(item) for item in value)
        return value

    temp_parent = "/tmp/opencode" if os.path.isdir("/tmp/opencode") else None
    with tempfile.TemporaryDirectory(prefix="m6_recurrent_", dir=temp_parent) as temp_dir:
        checkpoint = os.path.join(temp_dir, "model_audit.pt")
        runner.current_learning_iteration = 3
        print(f"[INFO] Saving temporary checkpoint to {checkpoint}.", flush=True)
        saved_dict = {
            "model_state_dict": to_cpu(runner.alg.policy.state_dict()),
            "optimizer_state_dict": to_cpu(runner.alg.optimizer.state_dict()),
            "iter": runner.current_learning_iteration,
            "infos": {"audit": "m6_recurrent_runtime"},
        }
        torch.save(saved_dict, checkpoint)
        print("[INFO] Constructing runner for checkpoint reload.", flush=True)
        reloaded = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print("[INFO] Loading temporary checkpoint.", flush=True)
        infos = reloaded.load(checkpoint, map_location=agent_cfg.device)
        _assert(reloaded.current_learning_iteration == 3, "Checkpoint iteration was not restored.")
        _assert(infos == {"audit": "m6_recurrent_runtime"}, "Checkpoint infos were not restored.")
        return {"checkpoint_loaded": True, "iteration": int(reloaded.current_learning_iteration)}


def _rollout(runner: OnPolicyRunner, rsl_env: RslRlVecEnvWrapper, steps: int) -> dict[str, Any]:
    policy = runner.alg.policy
    policy.reset()
    obs = rsl_env.get_observations().to(runner.device)
    _assert_finite("reset_obs", obs)
    done_count = 0
    reward_sum = 0.0
    for _ in range(steps):
        with torch.inference_mode():
            actions = policy.act_inference(obs)
            obs, rewards, dones, _ = rsl_env.step(actions.to(rsl_env.device))
            policy.reset(dones.to(device=runner.device))
            obs = obs.to(runner.device)
        _assert_finite("obs", obs)
        _assert_finite("rewards", rewards)
        done_count += int(torch.count_nonzero(dones).item())
        reward_sum += float(torch.sum(rewards).item())
    policy.reset()
    return {"steps": int(steps), "done_count": done_count, "reward_sum": reward_sum}


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
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
        print("[INFO] Creating M6 gym environment.", flush=True)
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        print("[INFO] Wrapping environment for RSL-RL.", flush=True)
        rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        print("[INFO] Constructing OnPolicyRunner.", flush=True)
        runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print("[INFO] Auditing recurrent policy contract.", flush=True)
        contract = _assert_policy_contract(runner)
        print("[INFO] Auditing checkpoint save/load.", flush=True)
        checkpoint = _audit_checkpoint(runner, rsl_env, agent_cfg)
        obs = rsl_env.get_observations().to(agent_cfg.device)
        print("[INFO] Auditing hidden-state reset.", flush=True)
        hidden_reset = _audit_hidden_reset(runner, obs)
        print("[INFO] Running short recurrent rollout.", flush=True)
        rollout = _rollout(runner, rsl_env, args_cli.steps)
        task = gym_env.unwrapped
        report = {
            "task": args_cli.task,
            "seed": int(args_cli.seed),
            "num_envs": int(args_cli.num_envs),
            "contract": contract,
            "hidden_reset": hidden_reset,
            "checkpoint": checkpoint,
            "rollout": rollout,
            "mode_counts": task.target_motion_manager.mode_counts(),
        }
        print(f"[INFO] M6 recurrent runtime audit: {json.dumps(report, sort_keys=True)}", flush=True)
    finally:
        if rsl_env is not None:
            rsl_env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
