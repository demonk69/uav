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
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7a_env_cfg.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/observations/__init__.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/observations/configs.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/observations/history_buffer.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/observations/corruption.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/observations/pipeline.py",
        "scripts/audit_m2_runtime.py",
        "scripts/audit_m3_motion_runtime.py",
        "scripts/audit_m4_baseline_runtime.py",
        "scripts/audit_m5_rl_runtime.py",
        "scripts/audit_m6_recurrent_runtime.py",
        "scripts/audit_m6_history_sensitivity.py",
        "scripts/audit_m6_checkpoint_resume.py",
        "scripts/audit_m7_observation_pipeline.py",
        "scripts/audit_m7_pomdp_comparison.py",
        "scripts/audit_m7b_dynamics.py",
        "scripts/evaluate_m7b_robustness.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/__init__.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/stateless_rng.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/action_delay.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/m7b.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/dynamics/wind.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7b_env.py",
        "source/uav_rendezvous_rl/uav_rendezvous_rl/tasks/direct/uav_rendezvous_m7b_env_cfg.py",
        "tests/test_m7b_action_delay.py",
        "tests/test_m7b_actor_isolation.py",
        "tests/test_m7b_critic_layout.py",
        "tests/test_m7b_dynamics_equation.py",
        "tests/test_m7b_fair_ablation.py",
        "tests/test_m7b_noncontact_objective.py",
        "tests/test_m7b_parameter_sampling.py",
        "tests/test_m7b_partial_reset.py",
        "tests/test_m7b_rng_reproducibility.py",
        "tests/test_m7b_task_registration.py",
        "tests/test_m7b_wind_process.py",
    ]

    for relative_path in required_files:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path


def test_milestone_state_authorizes_m7b_s1_only() -> None:
    text = (PROJECT_ROOT / "docs/milestone_state.md").read_text(encoding="utf-8")

    assert "Current milestone: M7" in text
    assert "Current sub-milestone: M7B" in text
    assert "Status: in_progress" in text
    assert "Next executable stage: M7B-S1" in text
    assert "M7C authorization: not authorized" in text
    assert "Authoritative M7B execution order: M7B-S0 clean, M7B-S1 dynamics only" in text
    assert "Current executable stage: M7B-S1 dynamics only" in text
    assert "Crazyflie" in text
