"""M3 smoke tests that do not launch Isaac Sim."""

import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_required_m3_files_exist() -> None:
    required_files = [
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "docs/environment_audit.md",
        "docs/implementation_plan.md",
        "docs/milestone_state.md",
        "scripts/zero_agent.py",
        "scripts/random_agent.py",
        "source/uav_rendezvous_rl/setup.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/__init__.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/base.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/configs.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/constant_velocity.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/constant_acceleration.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/constant_turn.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/piecewise_acceleration.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/manager.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/motions/sampling.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/m2_kinematics.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env_cfg.py",
        "scripts/audit_m2_runtime.py",
        "scripts/audit_m3_motion_runtime.py",
    ]

    for relative_path in required_files:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path


def test_milestone_state_forbids_m4_work() -> None:
    text = (PROJECT_ROOT / "docs/milestone_state.md").read_text(encoding="utf-8")

    assert "Current milestone: M3" in text
    assert "Next milestone: M4, not authorized" in text
    assert "ConstantAcceleration" in text
    assert "PPO训练" in text
    assert "Crazyflie" in text
