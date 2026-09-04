"""Bounded-policy and audit-log safety checks for the Phase 3 agent."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "audit"))

from logger import AUDIT_PATH  # noqa: E402
from policies import MAX_RETRY_ATTEMPTS, decide  # noqa: E402

REQUIRED_AUDIT_FIELDS = (
    "transaction_id",
    "predicted_root_cause",
    "model_recommended_action",
    "override_fired",
    "final_action_taken",
    "policy_reason",
    "action_result",
    "amount",
    "amount_recovered",
    "retry_attempt_number",
    "timestamp",
)


def load_audit() -> list[dict]:
    assert AUDIT_PATH.is_file(), f"missing {AUDIT_PATH}; run agent/orchestrator.py"
    rows = []
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    assert rows, "audit_log.jsonl is empty"
    return rows


@pytest.fixture(scope="module")
def audit_rows() -> list[dict]:
    return load_audit()


def test_no_retry_attempt_exceeds_hard_cap(audit_rows):
    over = [
        r
        for r in audit_rows
        if int(r["retry_attempt_number"]) > MAX_RETRY_ATTEMPTS
    ]
    assert not over, (
        f"{len(over)} rows have retry_attempt_number > {MAX_RETRY_ATTEMPTS}"
    )


def test_every_risk_decline_is_overridden_to_escalate(audit_rows):
    risk = [r for r in audit_rows if r.get("failure_code") == "risk_decline"]
    assert risk, "holdout audit log has no risk_decline rows"
    failures = []
    for r in risk:
        if r.get("override_fired") is not True:
            failures.append((r["transaction_id"], "override_fired", r.get("override_fired")))
        if r.get("final_action_taken") != "ESCALATE_HUMAN":
            failures.append(
                (r["transaction_id"], "final_action_taken", r.get("final_action_taken"))
            )
    assert not failures, f"risk_decline safety failures: {failures}"


def test_safety_override_changes_a_retry_recommendation():
    """Classifier/map can recommend retry; raw risk_decline must still escalate."""
    decision = decide(
        {
            "failure_code": "risk_decline",
            "retry_count_so_far": 0,
            "failure_timestamp": "2020-01-01T00:00:00Z",
            "amount": 100.0,
        },
        "RETRY_IMMEDIATE",
    )
    assert decision.override_fired is True
    assert decision.final_action == "ESCALATE_HUMAN"
    assert decision.execute_retry is False
    assert "RETRY_IMMEDIATE" != decision.final_action


def test_override_fired_is_not_a_noop_flag(audit_rows):
    """When override_fired, policy must change the action unless the model
    already recommended ESCALATE_HUMAN (defense-in-depth on risk_decline).
    """
    fired = [r for r in audit_rows if r.get("override_fired") is True]
    assert fired, "override_fired was never True (flag is a no-op)"
    for r in fired:
        model_a = r["model_recommended_action"]
        final_a = r["final_action_taken"]
        if r.get("failure_code") == "risk_decline" and model_a == "ESCALATE_HUMAN":
            assert final_a == "ESCALATE_HUMAN"
            continue
        assert model_a != final_a, (
            f"{r['transaction_id']}: override_fired but "
            f"model_recommended_action == final_action_taken == {model_a}"
        )


def test_every_row_has_nonempty_policy_reason(audit_rows):
    missing = [
        r["transaction_id"]
        for r in audit_rows
        if not str(r.get("policy_reason") or "").strip()
    ]
    assert not missing, f"empty policy_reason for {missing}"


def test_retry_attempt_number_monotonic_per_transaction(audit_rows):
    by_tx: dict[str, list[dict]] = defaultdict(list)
    for r in audit_rows:
        by_tx[str(r["transaction_id"])].append(r)
    for tx_id, group in by_tx.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: str(r.get("timestamp") or ""))
        attempts = [int(r["retry_attempt_number"]) for r in ordered]
        assert attempts == sorted(attempts), (
            f"{tx_id} retry_attempt_number not monotonic: {attempts}"
        )


def test_audit_rows_have_required_fields(audit_rows):
    for r in audit_rows:
        missing = [k for k in REQUIRED_AUDIT_FIELDS if k not in r]
        assert not missing, f"{r.get('transaction_id')} missing {missing}"
