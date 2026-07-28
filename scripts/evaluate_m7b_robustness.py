#!/usr/bin/env python3
"""Launch independent-process M7B robustness evaluations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_SH = Path("/home/lab_726/IsaacLab/isaaclab.sh")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent-process M7B robustness evaluation launcher.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path to evaluate. May be repeated.")
    parser.add_argument("--task", action="append", default=[], help="Task ID for each checkpoint. Defaults to M7B feedforward.")
    parser.add_argument("--m7b_stage", action="append", default=[], help="M7B stage for each evaluation. Defaults to 4.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--policy", choices=("trained", "zero", "random", "oracle"), default="trained")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print commands without executing them.")
    return parser


def _cleared_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return env


def _items(values: list[str], count: int, default: str) -> list[str]:
    if not values:
        return [default for _ in range(count)]
    if len(values) != count:
        raise RuntimeError(f"Expected {count} values, got {len(values)}.")
    return values


def main() -> None:
    args = _build_parser().parse_args()
    count = max(1, len(args.checkpoint))
    checkpoints = _items(args.checkpoint, count, "")
    tasks = _items(args.task, count, "Isaac-Uav-Rendezvous-M7B-Feedforward-v0")
    stages = _items(args.m7b_stage, count, "4")
    env = _cleared_env()
    for checkpoint, task, stage in zip(checkpoints, tasks, stages, strict=True):
        command = [
            str(ISAACLAB_SH),
            "-p",
            str(PROJECT_ROOT / "scripts/evaluate.py"),
            "--task",
            task,
            "--m7b_stage",
            stage,
            "--num_envs",
            str(args.num_envs),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--split",
            args.split,
            "--policy",
            args.policy,
            "--headless",
        ]
        if checkpoint:
            command.extend(("--checkpoint", checkpoint))
        print("[INFO] launching independent M7B evaluation:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
