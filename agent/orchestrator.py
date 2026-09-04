"""Run classifier -> policy -> bounded action over the holdout batch."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "actions"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "audit"))

from escalate_human import escalate_human  # noqa: E402
from intervention_map import recommend_action  # noqa: E402
from logger import AUDIT_PATH, log_decision, read_log, reset_log  # noqa: E402
from policies import MAX_RETRY_ATTEMPTS, RETRY_ACTIONS, decide  # noqa: E402
from retry_mandate import retry_mandate  # noqa: E402
from retry_payment import retry_payment  # noqa: E402
from send_nudge import send_nudge  # noqa: E402
from train_classifier import load_bundle  # noqa: E402

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_set.csv"
TARGET = "true_root_cause_category"
RETRY_GATEWAY_FEE_INR = 5.0


def _no_action(transaction: dict[str, Any], detail: str) -> dict[str, Any]:
    return {"status": "no_action", "detail": detail, "amount_recovered": 0.0}


def execute_action(
    transaction: dict[str, Any],
    decision,
) -> dict[str, Any]:
    action = decision.final_action
    if decision.execute_retry:
        if (
            transaction.get("failure_code") == "expired_mandate"
            or transaction.get("predicted_root_cause") == "MANDATE_EXPIRED"
        ):
            return retry_mandate(transaction)
        return retry_payment(transaction)
    if action in RETRY_ACTIONS and not decision.execute_retry:
        return {
            "status": "deferred",
            "detail": decision.policy_reason,
            "amount_recovered": 0.0,
        }
    if action == "SEND_NUDGE":
        return send_nudge(transaction)
    if action == "ESCALATE_HUMAN":
        return escalate_human(transaction)
    if action == "NO_ACTION_UNRECOVERABLE":
        return _no_action(
            transaction,
            "Marked unrecoverable; skipping automated retry and nudge.",
        )
    return _no_action(transaction, f"Unhandled action {action}; no side effect.")


def predict_batch(holdout: pd.DataFrame) -> pd.Series:
    bundle = load_bundle()
    encoded = bundle["model"].predict(bundle["transformer"].transform(holdout))
    labels = bundle["label_encoder"].inverse_transform(encoded)
    return pd.Series(labels, index=holdout.index, name="predicted_root_cause")


def process_transaction(row: dict[str, Any]) -> dict[str, Any]:
    predicted = str(row["predicted_root_cause"])
    model_action = recommend_action(predicted)
    decision = decide(row, model_action)
    # Stash prediction so mandate routing can see it.
    row = dict(row)
    row["predicted_root_cause"] = predicted
    result = execute_action(row, decision)
    record = {
        "transaction_id": row.get("transaction_id"),
        "predicted_root_cause": predicted,
        "model_recommended_action": model_action,
        "override_fired": decision.override_fired,
        "final_action_taken": decision.final_action,
        "policy_reason": decision.policy_reason,
        "action_result": result,
        "amount": float(row.get("amount") or 0),
        "amount_recovered": float(result.get("amount_recovered") or 0),
        "retry_attempt_number": decision.retry_attempt_number,
        "true_root_cause_category": row.get(TARGET),
        "failure_code": row.get("failure_code"),
        "execute_retry": decision.execute_retry,
    }
    log_decision(record)
    return record


def wasted_retry_cost(records: list[dict[str, Any]]) -> tuple[float, int]:
    """INR 5 per executed retry on payments that were never recovered."""
    attempts = 0
    for rec in records:
        if not rec.get("execute_retry"):
            continue
        if rec.get("final_action_taken") not in RETRY_ACTIONS:
            continue
        if float(rec.get("amount_recovered") or 0) > 0:
            continue
        attempts += 1
    return attempts * RETRY_GATEWAY_FEE_INR, attempts


def print_batch_checks(records: list[dict[str, Any]]) -> None:
    n = len(records)
    ids = [r["transaction_id"] for r in records]
    print(f"\n=== Batch checks ===")
    print(f"audit_log.jsonl lines: {n} (expected 60 unique holdout rows)")
    print(f"unique transaction_id: {len(set(ids))}")

    over_cap = [r for r in records if int(r["retry_attempt_number"]) > MAX_RETRY_ATTEMPTS]
    print(f"rows with retry_attempt_number > {MAX_RETRY_ATTEMPTS}: {len(over_cap)}")

    risk = [r for r in records if r.get("failure_code") == "risk_decline"]
    risk_retried = [
        r
        for r in risk
        if r.get("execute_retry") or r.get("final_action_taken") in RETRY_ACTIONS
    ]
    risk_override = all(bool(r.get("override_fired")) for r in risk) if risk else False
    print(f"risk_decline rows: {len(risk)}")
    print(f"risk_decline override_fired all true: {risk_override}")
    print(f"risk_decline retried: {len(risk_retried)}")

    counts = Counter(r["final_action_taken"] for r in records)
    print("\n=== Actions taken by type ===")
    for action, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {action}: {count}")

    recovered = sum(float(r["amount_recovered"]) for r in records)
    print(f"\nTotal amount_recovered: INR {recovered:,.2f}")
    cost, n_wasted = wasted_retry_cost(records)
    print(
        f"Wasted retry cost (retry attempts on transactions never recovered, "
        f"INR {RETRY_GATEWAY_FEE_INR:.0f}/attempt): "
        f"INR {cost:,.2f}  ({n_wasted} attempts)"
    )


def main() -> None:
    holdout = pd.read_csv(HOLDOUT_PATH)
    holdout["predicted_root_cause"] = predict_batch(holdout)
    reset_log()
    records = []
    for row in holdout.to_dict(orient="records"):
        records.append(process_transaction(row))
    logged = read_log()
    if len(logged) != len(holdout):
        raise SystemExit(
            f"Audit log has {len(logged)} lines, holdout has {len(holdout)} rows"
        )
    print_batch_checks(logged)
    print(f"\nWrote {AUDIT_PATH}")


if __name__ == "__main__":
    main()
