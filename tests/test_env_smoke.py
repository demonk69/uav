"""M2 smoke tests that do not launch Isaac Sim."""

import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_required_m2_files_exist() -> None:
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
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/m2_kinematics.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_env_cfg.py",
    ]

    for relative_path in required_files:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path


def test_milestone_state_forbids_m3_work() -> None:
    text = (PROJECT_ROOT / "docs/milestone_state.md").read_text(encoding="utf-8")

    assert "Current milestone: M2" in text
    assert "Next milestone: M3, not authorized" in text
    assert "ConstantAcceleration" in text
    assert "PPO训练" in text
    assert "Crazyflie" in text
