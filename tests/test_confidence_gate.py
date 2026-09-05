"""Tests for the low-confidence safety gate in agent/policies.py.

Coverage
--------
1. Low confidence → ESCALATE_HUMAN with low_confidence_override_fired=True.
2. risk_decline wins over low confidence when both rules apply.
3. High confidence, no risk_decline → both override flags False (normal path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "model"))

from policies import CONFIDENCE_THRESHOLD, decide  # noqa: E402
from predict import CONFIDENCE_THRESHOLD as PREDICT_THRESHOLD  # noqa: E402


# ── Shared fixture data ────────────────────────────────────────────────────────

def _txn(
    failure_code: str = "bank_timeout",
    confidence: float = 0.9,
    retry_count_so_far: int = 0,
) -> dict:
    return {
        "failure_code": failure_code,
        "classifier_confidence": confidence,
        "retry_count_so_far": retry_count_so_far,
        "failure_timestamp": None,
        "amount": 1000.0,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_confidence_threshold_is_shared():
    """CONFIDENCE_THRESHOLD imported in policies must equal predict.py's value."""
    assert CONFIDENCE_THRESHOLD == PREDICT_THRESHOLD, (
        "policies.CONFIDENCE_THRESHOLD and predict.CONFIDENCE_THRESHOLD diverged. "
        "Both must reference the same constant."
    )


def test_low_confidence_escalates_to_human():
    """A transaction with top-class probability below the gate threshold
    must be escalated to ESCALATE_HUMAN with low_confidence_override_fired=True,
    regardless of what the intervention map recommends.
    """
    low_prob = CONFIDENCE_THRESHOLD - 0.15   # well below threshold
    txn = _txn(failure_code="bank_timeout", confidence=low_prob)

    # Intervention map would normally say RETRY_IMMEDIATE for TEMPORARY_BANK_ISSUE
    decision = decide(txn, "RETRY_IMMEDIATE")

    assert decision.final_action == "ESCALATE_HUMAN", (
        f"Expected ESCALATE_HUMAN for confidence={low_prob}, "
        f"got {decision.final_action}"
    )
    assert decision.low_confidence_override_fired is True
    assert decision.override_fired is False, (
        "override_fired should stay False when the confidence gate fires "
        "(it is reserved for risk_decline and retry-cap overrides)"
    )
    assert decision.execute_retry is False
    assert str(low_prob) in decision.policy_reason or "LOW CONFIDENCE" in decision.policy_reason


def test_low_confidence_wins_over_retry_recommendation():
    """Low confidence should escalate even when the model recommends a nudge."""
    low_prob = CONFIDENCE_THRESHOLD - 0.10
    txn = _txn(failure_code="insufficient_funds", confidence=low_prob)

    decision = decide(txn, "SEND_NUDGE")

    assert decision.final_action == "ESCALATE_HUMAN"
    assert decision.low_confidence_override_fired is True
    assert decision.override_fired is False


def test_risk_decline_wins_over_low_confidence():
    """When BOTH risk_decline and low confidence would fire on the same
    transaction, risk_decline takes priority:
      override_fired=True, low_confidence_override_fired=False.
    """
    low_prob = CONFIDENCE_THRESHOLD - 0.20   # would normally trigger confidence gate
    txn = _txn(failure_code="risk_decline", confidence=low_prob)

    decision = decide(txn, "RETRY_IMMEDIATE")

    assert decision.final_action == "ESCALATE_HUMAN"
    assert decision.override_fired is True, (
        "risk_decline override must set override_fired=True"
    )
    assert decision.low_confidence_override_fired is False, (
        "low_confidence_override_fired must be False when risk_decline fires "
        "(risk_decline is priority-1 and checked first)"
    )
    assert decision.execute_retry is False
    assert "risk_decline" in decision.policy_reason.lower()


def test_high_confidence_no_override_flags():
    """High confidence, non-risk_decline → both override flags are False."""
    high_prob = CONFIDENCE_THRESHOLD + 0.30
    txn = _txn(failure_code="bank_timeout", confidence=high_prob)

    decision = decide(txn, "RETRY_IMMEDIATE")

    assert decision.override_fired is False
    assert decision.low_confidence_override_fired is False


def test_exactly_at_threshold_is_not_escalated():
    """A confidence equal to CONFIDENCE_THRESHOLD should NOT trigger the gate
    (gate fires on strict less-than: confidence < CONFIDENCE_THRESHOLD).
    """
    txn = _txn(failure_code="bank_timeout", confidence=CONFIDENCE_THRESHOLD)

    decision = decide(txn, "RETRY_IMMEDIATE")

    assert decision.low_confidence_override_fired is False, (
        f"At exactly the threshold ({CONFIDENCE_THRESHOLD}) the gate must NOT fire"
    )


def test_confidence_gate_reason_contains_threshold():
    """The policy_reason for a low-confidence escalation must mention the threshold
    so operators can diagnose it from the audit log.
    """
    low_prob = CONFIDENCE_THRESHOLD - 0.05
    txn = _txn(confidence=low_prob)
    decision = decide(txn, "RETRY_IMMEDIATE")

    assert decision.low_confidence_override_fired is True
    reason = decision.policy_reason
    assert "LOW CONFIDENCE" in reason or "low_confidence" in reason.lower() or str(CONFIDENCE_THRESHOLD) in reason, (
        f"policy_reason does not mention the confidence gate or threshold: {reason!r}"
    )
