"""Run classifier -> policy -> bounded action over the holdout batch."""

from __future__ import annotations

import argparse
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


def _as_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def predict_with_confidence(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    bundle = load_bundle()
    proba = bundle["model"].predict_proba(bundle["transformer"].transform(frame))
    encoded = proba.argmax(axis=1)
    labels = bundle["label_encoder"].inverse_transform(encoded)
    return (
        pd.Series(labels, index=frame.index, name="predicted_root_cause"),
        pd.Series(proba.max(axis=1), index=frame.index, name="classifier_confidence"),
    )


def predict_batch(holdout: pd.DataFrame) -> pd.Series:
    labels, _confidence = predict_with_confidence(holdout)
    return labels


def predict_one(row: dict[str, Any]) -> tuple[str, float]:
    frame = pd.DataFrame([row])
    labels, confidence = predict_with_confidence(frame)
    return str(labels.iloc[0]), float(confidence.iloc[0])


def process_transaction(row: dict[str, Any]) -> dict[str, Any]:
    row = {key: _as_python(value) for key, value in dict(row).items()}
    predicted = row.get("predicted_root_cause")
    confidence = row.get("classifier_confidence")
    if predicted is None or predicted == "" or confidence is None:
        predicted, confidence = predict_one(row)
    predicted = str(predicted)
    confidence = float(confidence)
    model_action = recommend_action(predicted)
    decision = decide(row, model_action)
    # Stash prediction so mandate routing can see it.
    row["predicted_root_cause"] = predicted
    result = execute_action(row, decision)
    record = {
        "transaction_id": str(row.get("transaction_id") or ""),
        "predicted_root_cause": predicted,
        "classifier_confidence": round(confidence, 4),
        "model_recommended_action": model_action,
        "override_fired": decision.override_fired,
        "final_action_taken": decision.final_action,
        "policy_reason": decision.policy_reason,
        "action_result": result,
        "amount": float(row.get("amount") or 0),
        "amount_recovered": float(result.get("amount_recovered") or 0),
        "retry_attempt_number": int(decision.retry_attempt_number),
        "true_root_cause_category": row.get(TARGET),
        "failure_code": row.get("failure_code"),
        "execute_retry": bool(decision.execute_retry),
        "retry_cap": MAX_RETRY_ATTEMPTS,
    }
    log_decision(record)
    return record


def load_holdout_row(transaction_id: str) -> dict[str, Any]:
    holdout = pd.read_csv(HOLDOUT_PATH)
    matched = holdout[holdout["transaction_id"].astype(str) == str(transaction_id)]
    if matched.empty:
        raise KeyError(f"transaction_id {transaction_id!r} is not in {HOLDOUT_PATH}")
    return {key: _as_python(value) for key, value in matched.iloc[0].to_dict().items()}


def process_transaction_id(transaction_id: str) -> dict[str, Any]:
    """Classify and act on one holdout row; append to the audit log (no reset)."""
    row = load_holdout_row(transaction_id)
    predicted, confidence = predict_one(row)
    row["predicted_root_cause"] = predicted
    row["classifier_confidence"] = confidence
    return process_transaction(row)


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Revenue recovery orchestrator")
    parser.add_argument(
        "--transaction-id",
        help="Process a single holdout transaction and append to the audit log",
    )
    args = parser.parse_args(argv)

    if args.transaction_id:
        record = process_transaction_id(args.transaction_id)
        print(
            f"{record['transaction_id']}  {record['predicted_root_cause']}  "
            f"{record['final_action_taken']}  recovered={record['amount_recovered']:.2f}"
        )
        print(f"Appended {AUDIT_PATH}")
        return

    holdout = pd.read_csv(HOLDOUT_PATH)
    labels, confidence = predict_with_confidence(holdout)
    holdout["predicted_root_cause"] = labels
    holdout["classifier_confidence"] = confidence
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
