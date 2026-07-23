"""Smoke tests that do not launch Isaac Sim."""

import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_required_m5_files_exist() -> None:
    required_files = [
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "docs/environment_audit.md",
        "docs/implementation_plan.md",
        "docs/milestone_state.md",
        "scripts/zero_agent.py",
        "scripts/random_agent.py",
        "scripts/train.py",
        "scripts/play.py",
        "scripts/evaluate.py",
        "source/uav_rendezvous_rl/setup.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/mdp/__init__.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/mdp/rendezvous.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/__init__.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/base.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/configs.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/constant_velocity.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/constant_acceleration.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/constant_turn.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/piecewise_acceleration.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/manager.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/sampling.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/controllers/__init__.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/controllers/baseline.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/controllers/configs.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/m4_accounting.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/m2_kinematics.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env_cfg.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_baseline_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_baseline_env_cfg.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_rl_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_rl_env_cfg.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_recurrent_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_recurrent_env_cfg.py",
        "scripts/audit_m2_runtime.py",
        "scripts/audit_m3_motion_runtime.py",
        "scripts/audit_m4_baseline_runtime.py",
        "scripts/audit_m5_rl_runtime.py",
        "scripts/audit_m6_recurrent_runtime.py",
        "scripts/audit_m6_history_sensitivity.py",
        "scripts/audit_m6_checkpoint_resume.py",
    ]

    for relative_path in required_files:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path


def test_milestone_state_authorizes_only_m6_work() -> None:
    text = (PROJECT_ROOT / "docs/milestone_state.md").read_text(encoding="utf-8")

    assert "Current milestone: M6" in text
    assert "Next milestone: M7, not authorized" in text
    assert "独立Recurrent RL任务" in text
    assert "GRU Recurrent PPO" in text
    assert "公平前馈ablation对照" in text
    assert "Crazyflie" in text
