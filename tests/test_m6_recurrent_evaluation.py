"""Static tests for recurrent-aware play/evaluate inference loops."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
PLAY_SOURCE = ROOT / "scripts/play.py"
EVALUATE_SOURCE = ROOT / "scripts/evaluate.py"


def test_play_uses_recurrent_safe_inference_and_done_reset() -> None:
    text = PLAY_SOURCE.read_text(encoding="utf-8")

    assert "policy_nn.act_inference" in text
    assert "policy_nn.actor(actor_obs)" not in text
    assert "_reset_policy(policy_nn, dones)" in text
    assert "--audit_hidden_state" in text


def test_evaluate_reports_per_mode_metrics_and_determinism_check() -> None:
    text = EVALUATE_SOURCE.read_text(encoding="utf-8")

    assert "policy_nn.act_inference" in text
    assert "policy_nn.actor(actor_obs)" not in text
    assert "_reset_policy(policy_nn, dones)" in text
    assert "summary_by_mode" in text
    assert "--determinism_check" in text
