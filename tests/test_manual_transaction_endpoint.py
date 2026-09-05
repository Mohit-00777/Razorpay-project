"""Tests for the /manual-transaction endpoint.

Coverage
--------
1. Dry-run POST → audit_log.jsonl line count is UNCHANGED.
2. Live POST → audit_log.jsonl gains EXACTLY ONE new line.
3. Dry-run response contains simulation_only=True and full reasoning fields.
4. Live response contains standard audit fields (no simulation_only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]

# Ensure all project sub-packages are importable before importing the app.
for _p in (
    ROOT / "agent",
    ROOT / "agent" / "actions",
    ROOT / "model",
    ROOT / "audit",
    ROOT / "dashboard",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from dashboard.app import app  # noqa: E402

AUDIT_PATH = ROOT / "audit" / "audit_log.jsonl"

_SAMPLE_PAYLOAD = {
    "amount": 1500.0,
    "failure_code": "bank_timeout",
    "payment_method": "card",
    "retry_count_so_far": 0,
    "bank_name": "HDFC",
}


def _count_audit_lines() -> int:
    """Return the current number of non-empty lines in audit_log.jsonl."""
    if not AUDIT_PATH.exists():
        return 0
    return sum(
        1
        for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ── Dry-run tests ──────────────────────────────────────────────────────────────

def test_dry_run_does_not_write_audit_log(client: TestClient):
    """POSTing with dry_run=True must NOT append to audit_log.jsonl."""
    before = _count_audit_lines()

    resp = client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": True})
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"

    after = _count_audit_lines()
    assert after == before, (
        f"audit_log.jsonl grew by {after - before} lines during a dry-run request. "
        "Dry-run must NOT write to the audit log."
    )


def test_dry_run_response_has_simulation_flag(client: TestClient):
    """Dry-run response must carry simulation_only=True so callers can distinguish it."""
    resp = client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("simulation_only") is True or data.get("dry_run") is True, (
        f"simulation_only / dry_run flag missing from response: {data}"
    )


def test_dry_run_response_has_full_reasoning_fields(client: TestClient):
    """Dry-run response must include all fields needed to explain the decision."""
    resp = client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": True})
    assert resp.status_code == 200
    data = resp.json()
    required = [
        "predicted_root_cause",
        "classifier_confidence",
        "proba_breakdown",
        "model_recommended_action",
        "override_fired",
        "low_confidence_override_fired",
        "final_action_taken",
        "policy_reason",
    ]
    missing = [k for k in required if k not in data]
    assert not missing, f"Dry-run response missing fields: {missing}\nFull response: {data}"


def test_dry_run_proba_breakdown_sums_to_one(client: TestClient):
    """The proba_breakdown dict probabilities must sum to approximately 1.0."""
    resp = client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": True})
    assert resp.status_code == 200
    breakdown = resp.json().get("proba_breakdown", {})
    assert breakdown, "proba_breakdown is empty"
    total = sum(breakdown.values())
    assert abs(total - 1.0) < 0.01, f"proba_breakdown sums to {total:.4f}, expected ~1.0"


# ── Live-run tests ─────────────────────────────────────────────────────────────

def test_live_run_writes_exactly_one_audit_line(client: TestClient):
    """POSTing with dry_run=False must append EXACTLY ONE line to audit_log.jsonl."""
    before = _count_audit_lines()

    resp = client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": False})
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"

    after = _count_audit_lines()
    delta = after - before
    assert delta == 1, (
        f"Expected audit_log.jsonl to gain exactly 1 line for a live run, "
        f"but it changed by {delta} (before={before}, after={after})."
    )


def test_live_run_response_is_not_simulation(client: TestClient):
    """Live response must NOT carry simulation_only=True."""
    resp = client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": False})
    assert resp.status_code == 200
    data = resp.json()
    assert not data.get("simulation_only"), (
        f"Live run should not set simulation_only, got: {data.get('simulation_only')}"
    )


def test_live_run_audit_line_is_valid_json(client: TestClient):
    """The last appended audit line must be parseable and contain required fields."""
    client.post("/manual-transaction", json={**_SAMPLE_PAYLOAD, "dry_run": False})

    lines = [
        line.strip()
        for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    last_line = lines[-1]
    record = json.loads(last_line)  # raises if malformed

    for field in ("transaction_id", "predicted_root_cause", "classifier_confidence",
                  "final_action_taken", "policy_reason", "override_fired",
                  "low_confidence_override_fired", "timestamp"):
        assert field in record, f"Audit record missing field {field!r}: {record}"


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_risk_decline_dry_run_escalates_without_logging(client: TestClient):
    """risk_decline failure code should escalate even in dry-run,
    and must not write to the audit log.
    """
    before = _count_audit_lines()
    payload = {**_SAMPLE_PAYLOAD, "failure_code": "risk_decline", "dry_run": True}
    resp = client.post("/manual-transaction", json=payload)
    assert resp.status_code == 200

    after = _count_audit_lines()
    assert after == before, "Dry-run risk_decline must not write to audit log"

    data = resp.json()
    assert data["final_action_taken"] == "ESCALATE_HUMAN"
    assert data["override_fired"] is True
    assert data["low_confidence_override_fired"] is False


def test_invalid_payload_returns_422(client: TestClient):
    """Missing required fields must return HTTP 422 (FastAPI validation)."""
    resp = client.post("/manual-transaction", json={"amount": -5})
    assert resp.status_code == 422
