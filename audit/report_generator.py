"""Summarize audit/audit_log.jsonl into a judge-facing final metrics report.

Usage
-----
    python audit/report_generator.py

Outputs
-------
  • Formatted summary printed to stdout
  • audit/final_report.json (machine-readable)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "audit", ROOT / "agent", ROOT / "model"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from logger import AUDIT_PATH, read_log  # noqa: E402
from orchestrator import wasted_retry_cost  # noqa: E402
from train_classifier import MODEL_PATH, load_bundle  # noqa: E402
from intervention_map import recommend_action  # noqa: E402

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_set.csv"
TARGET = "true_root_cause_category"
REPORT_PATH = Path(__file__).resolve().parent / "final_report.json"

UNRESOLVED_ACTIONS = {"ESCALATE_HUMAN", "NO_ACTION_UNRECOVERABLE"}

# INR gateway fee per executed retry attempt (from orchestrator constant)
RETRY_GATEWAY_FEE_INR = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def inr(value: float) -> str:
    return f"INR {value:,.2f}"


def latest_per_transaction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate: keep only the *last* audit record for each transaction_id."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = str(row.get("transaction_id") or "")
        latest[tid] = row
    return list(latest.values())


# ── Phase-2 false-positive cost ───────────────────────────────────────────────

def phase2_false_positive_cost() -> dict[str, Any]:
    """Compute holdout FP/FN rupee cost from the Phase-2 classifier evaluation."""
    if not MODEL_PATH.exists() or not HOLDOUT_PATH.exists():
        return {
            "available": False,
            "note": "Missing classifier.pkl or holdout_set.csv — run train_classifier.py first.",
            "misclassified_rows": 0,
            "fp_fn_cost_inr": 0.0,
            "recoverable_rupees_written_off_inr": 0.0,
        }

    bundle = load_bundle()
    holdout = pd.read_csv(HOLDOUT_PATH)
    y_true = holdout[TARGET]
    y_pred = pd.Series(
        bundle["label_encoder"].inverse_transform(
            bundle["model"].predict(bundle["transformer"].transform(holdout))
        ),
        index=holdout.index,
    )
    errors = holdout.loc[y_true != y_pred].copy()
    errors["predicted_root_cause"] = y_pred.loc[errors.index]
    errors["recommended_action"] = errors["predicted_root_cause"].map(recommend_action)

    # FP: classifier says PERMANENT_FAILURE but the true label is recoverable
    gave_up = (y_pred == "PERMANENT_FAILURE") & (y_true != "PERMANENT_FAILURE")

    return {
        "available": True,
        "source": "phase2_holdout_evaluation",
        "holdout_rows": int(len(holdout)),
        "misclassified_rows": int(len(errors)),
        "fp_fn_cost_inr": float(errors["amount"].sum()) if len(errors) else 0.0,
        "recoverable_rupees_written_off_inr": float(holdout.loc[gave_up, "amount"].sum()),
        "false_positive_detail": (
            "Rows where classifier predicted PERMANENT_FAILURE but true label "
            "was recoverable — money that would have been written off unnecessarily."
        ),
        "misclassified_transaction_ids": [
            str(tid) for tid in errors["transaction_id"].tolist()
        ],
    }


# ── Unresolved rows ────────────────────────────────────────────────────────────

def unresolved_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each un-recovered record by how it ended up open."""
    out: list[dict[str, Any]] = []
    for rec in records:
        if float(rec.get("amount_recovered") or 0) > 0:
            continue
        action = str(rec.get("final_action_taken") or "")
        status = str((rec.get("action_result") or {}).get("status") or "")

        is_escalated   = action in UNRESOLVED_ACTIONS or status == "escalated"
        is_retry_fail  = bool(rec.get("execute_retry")) and not (status == "captured")
        is_pending     = action == "SEND_NUDGE"
        is_deferred    = status == "deferred"

        if not (is_escalated or is_retry_fail or is_pending or is_deferred):
            continue

        if action == "ESCALATE_HUMAN" or status == "escalated":
            kind = "escalated"
        elif action == "NO_ACTION_UNRECOVERABLE":
            kind = "unrecoverable"
        elif is_pending:
            kind = "awaiting_customer"
        elif is_deferred:
            kind = "deferred_backoff"
        else:
            kind = "retry_failed"

        out.append(
            {
                "transaction_id":    rec.get("transaction_id"),
                "kind":              kind,
                "amount_inr":        float(rec.get("amount") or 0),
                "failure_code":      rec.get("failure_code"),
                "predicted_root_cause": rec.get("predicted_root_cause"),
                "final_action_taken": action,
                "reason":            rec.get("policy_reason"),
                "action_detail":     (rec.get("action_result") or {}).get("detail"),
            }
        )
    return out


# ── Build report dict ─────────────────────────────────────────────────────────

def build_report(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compute all metrics and return as a JSON-serialisable dict."""
    raw    = rows if rows is not None else read_log()
    unique = latest_per_transaction(raw)
    n      = len(unique)

    at_risk   = sum(float(r.get("amount") or 0)           for r in unique)
    recovered = sum(float(r.get("amount_recovered") or 0)  for r in unique)
    rate      = (recovered / at_risk * 100.0) if at_risk else 0.0

    actions = Counter(str(r.get("final_action_taken") or "UNKNOWN") for r in unique)

    # Recovered / not-recovered split
    n_recovered = sum(1 for r in unique if float(r.get("amount_recovered") or 0) > 0)

    waste_inr, waste_n = wasted_retry_cost(unique)
    unresolved = unresolved_rows(unique)
    kinds      = dict(Counter(item["kind"] for item in unresolved))

    # Per-action recovered amounts
    action_recovery: dict[str, dict[str, Any]] = {}
    for r in unique:
        act = str(r.get("final_action_taken") or "UNKNOWN")
        amt = float(r.get("amount_recovered") or 0)
        if act not in action_recovery:
            action_recovery[act] = {"count": 0, "recovered_inr": 0.0}
        action_recovery[act]["count"] += 1
        action_recovery[act]["recovered_inr"] += amt

    return {
        # ── Volume ──
        "audit_events":              len(raw),
        "total_transactions_processed": n,
        "transactions_recovered":    n_recovered,
        "transactions_not_recovered": n - n_recovered,

        # ── Money ──
        "total_amount_at_risk_inr":  round(at_risk, 2),
        "total_amount_recovered_inr": round(recovered, 2),
        "recovery_rate_pct":         round(rate, 2),

        # ── Actions ──
        "actions_taken":             dict(actions),
        "actions_with_recovery":     action_recovery,

        # ── Cost of wasted retries ──
        "wasted_retry_cost_inr":    round(waste_inr, 2),
        "wasted_retry_attempts":    waste_n,
        "retry_gateway_fee_per_attempt_inr": RETRY_GATEWAY_FEE_INR,

        # ── Phase-2 false-positive cost ──
        "false_positive_cost":      phase2_false_positive_cost(),

        # ── Override / gate breakdown ──
        "risk_decline_override_count": sum(
            1 for r in unique
            if r.get("override_fired") and not r.get("low_confidence_override_fired")
            and str(r.get("failure_code") or "") == "risk_decline"
        ),
        "retry_cap_override_count": sum(
            1 for r in unique
            if r.get("override_fired") and not r.get("low_confidence_override_fired")
            and str(r.get("failure_code") or "") != "risk_decline"
        ),
        "low_confidence_override_fired_count": sum(
            1 for r in unique if r.get("low_confidence_override_fired")
        ),
        "normal_path_count": sum(
            1 for r in unique
            if not r.get("override_fired") and not r.get("low_confidence_override_fired")
        ),

        # ── Unresolved ──
        "unresolved_or_escalated_count": len(unresolved),
        "unresolved_kind_breakdown":     kinds,
        "unresolved_or_escalated":       unresolved,

        # ── Meta ──
        "audit_path":  str(AUDIT_PATH),
        "report_path": str(REPORT_PATH),
    }


# ── Pretty-print ──────────────────────────────────────────────────────────────

_SEP  = "-" * 56
_SEP2 = "=" * 56


def print_report(report: dict[str, Any]) -> None:
    fp = report["false_positive_cost"]

    print(f"\n{_SEP2}")
    print("  Revenue Recovery Agent -- Final Metrics Report")
    print(f"{_SEP2}\n")

    print(f"  Audit events logged:      {report['audit_events']}")
    print(f"  Transactions processed:   {report['total_transactions_processed']}")
    print(f"  Transactions recovered:   {report['transactions_recovered']}")
    print(f"  Not recovered this cycle: {report['transactions_not_recovered']}")
    print()
    print(_SEP)
    print("  MONEY")
    print(_SEP)
    print(f"  At risk:            {inr(report['total_amount_at_risk_inr'])}")
    print(f"  Recovered:          {inr(report['total_amount_recovered_inr'])}")
    print(f"  Recovery rate:      {report['recovery_rate_pct']:.2f}%")
    print()
    print(_SEP)
    print("  ACTIONS TAKEN")
    print(_SEP)
    for action, meta in sorted(
        report["actions_with_recovery"].items(),
        key=lambda kv: (-kv[1]["count"], kv[0]),
    ):
        rec_inr = meta["recovered_inr"]
        rec_str = f"  (recovered {inr(rec_inr)})" if rec_inr > 0 else ""
        print(f"  {action:<30} x {meta['count']:>3}{rec_str}")

    print()
    print(_SEP)
    print("  COST OF WASTED RETRIES")
    print(_SEP)
    print(
        f"  INR {RETRY_GATEWAY_FEE_INR:.0f} / retry x "
        f"{report['wasted_retry_attempts']} failed attempts = "
        f"{inr(report['wasted_retry_cost_inr'])}"
    )

    print()
    print(_SEP)
    print("  FALSE-POSITIVE COST (Phase 2 holdout evaluation)")
    print(_SEP)
    if fp.get("available"):
        print(f"  Holdout rows evaluated:        {fp['holdout_rows']}")
        print(f"  Misclassified rows:            {fp['misclassified_rows']}")
        print(f"  FP + FN rupee exposure:        {inr(fp['fp_fn_cost_inr'])}")
        print(
            f"  Recoverable written off as     {inr(fp['recoverable_rupees_written_off_inr'])}"
        )
        print("    PERMANENT_FAILURE (false negatives)")
    else:
        print(f"  [!] {fp.get('note')}")

    print()
    print(_SEP)
    print("  DECISION PATH BREAKDOWN")
    print(_SEP)
    print(f"  risk_decline override:         {report.get('risk_decline_override_count', 0)}")
    print(f"  retry-cap override:            {report.get('retry_cap_override_count', 0)}")
    print(f"  low-confidence gate:           {report.get('low_confidence_override_fired_count', 0)}")
    print(f"  normal path (model + policy):  {report.get('normal_path_count', 0)}")

    print()
    print(_SEP)
    print(f"  UNRESOLVED / ESCALATED  ({report['unresolved_or_escalated_count']} total)")
    print(_SEP)
    for kind, count in sorted(
        report["unresolved_kind_breakdown"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {kind:<28} {count:>3}")

    explicit = [
        item
        for item in report["unresolved_or_escalated"]
        if item["kind"] in {"escalated", "unrecoverable"}
    ]
    if explicit:
        print()
        print("  Escalated / unrecoverable (explicit list):")
        for item in explicit:
            print(
                f"  - {item['transaction_id']}  "
                f"{inr(item['amount_inr'])}  [{item['kind']}]  "
                f"{item['final_action_taken']}"
            )
            reason = (item.get("reason") or "").split(".")[0][:90]
            print(f"    {reason}")

    print()
    print(_SEP)
    print(f"  Report saved -> {report['report_path']}")
    print(f"{_SEP2}\n")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    if not AUDIT_PATH.exists() or AUDIT_PATH.stat().st_size == 0:
        raise SystemExit(
            f"\n[!] No audit log at {AUDIT_PATH}.\n"
            "    Run `python agent/orchestrator.py` first to populate it.\n"
        )

    report = build_report()
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print_report(report)


if __name__ == "__main__":
    main()
