"""Static tests for M7B train/play/evaluate contract hooks."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_train_play_evaluate_accept_m7b_stage() -> None:
    for relative in ("scripts/train.py", "scripts/play.py", "scripts/evaluate.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "--m7b_stage" in text
        assert "_configure_m7b_stage" in text
        assert "tau_velocity_scale = 1.0" in text
        assert "action_delay_steps = 3" in text
        assert "gust_max_magnitude = 0.15" in text


def test_m7b_pomdp_comparison_script_is_deprecated() -> None:
    text = (ROOT / "scripts/audit_m7_pomdp_comparison.py").read_text(encoding="utf-8")

    assert "deprecated" in text
    assert "Same-process Isaac multi-environment lifecycle results are not trusted" in text
    assert "scripts/evaluate.py" in text
    assert "sys.exit(1)" in text


def test_m7b_runtime_audit_and_independent_evaluator_exist() -> None:
    audit = (ROOT / "scripts/audit_m7b_dynamics.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "scripts/evaluate_m7b_robustness.py").read_text(encoding="utf-8")

    assert "get_m7b_diagnostics" in audit
    assert "policy_class" in audit
    assert "partial_reset" in audit
    assert "subprocess.run" in evaluator
    assert "/home/lab_726/IsaacLab/isaaclab.sh" in evaluator
