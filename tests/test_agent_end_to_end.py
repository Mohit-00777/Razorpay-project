"""End-to-end checks against holdout_set.csv and audit/audit_log.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "audit"))

from logger import AUDIT_PATH  # noqa: E402
from orchestrator import RETRY_GATEWAY_FEE_INR, wasted_retry_cost  # noqa: E402

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_set.csv"
ACTION_RESULT_KEYS = {"status", "detail", "amount_recovered"}


def load_audit() -> list[dict]:
    rows = []
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def audit_rows() -> list[dict]:
    rows = load_audit()
    assert rows, "audit_log.jsonl is empty; run agent/orchestrator.py"
    return rows


@pytest.fixture(scope="module")
def holdout() -> pd.DataFrame:
    return pd.read_csv(HOLDOUT_PATH)


def test_one_audit_row_per_holdout_transaction(audit_rows, holdout):
    """Design: one JSONL line per holdout transaction (not per retry attempt).

    A retry this cycle is recorded as retry_attempt_number on that single line.
    """
    holdout_ids = set(holdout["transaction_id"].astype(str))
    audit_ids = [str(r["transaction_id"]) for r in audit_rows]
    assert len(audit_rows) == len(holdout), (
        f"audit has {len(audit_rows)} rows, holdout has {len(holdout)}"
    )
    assert len(audit_ids) == len(set(audit_ids)), "duplicate transaction_id in audit log"
    assert set(audit_ids) == holdout_ids


def test_action_result_has_required_keys(audit_rows):
    for r in audit_rows:
        result = r.get("action_result")
        assert isinstance(result, dict), f"{r['transaction_id']} action_result is not a dict"
        missing = ACTION_RESULT_KEYS - set(result)
        assert not missing, f"{r['transaction_id']} action_result missing {missing}"


def test_amount_recovered_is_sane(audit_rows, holdout):
    recovered = sum(float(r.get("amount_recovered") or 0) for r in audit_rows)
    batch_total = float(holdout["amount"].sum())
    assert recovered >= 0
    assert recovered <= batch_total + 1e-6


def test_wasted_retry_cost_matches_orchestrator_formula(audit_rows):
    derived_attempts = 0
    for r in audit_rows:
        result = r.get("action_result") or {}
        status = str(result.get("status") or "")
        recovered = float(r.get("amount_recovered") or 0)
        executed = bool(r.get("execute_retry"))
        if executed and recovered <= 0 and status != "captured":
            derived_attempts += 1
    derived_cost = derived_attempts * RETRY_GATEWAY_FEE_INR
    printed_cost, printed_attempts = wasted_retry_cost(audit_rows)
    assert derived_attempts == printed_attempts
    assert derived_cost == pytest.approx(printed_cost)
    assert derived_cost == pytest.approx(derived_attempts * 5.0)
