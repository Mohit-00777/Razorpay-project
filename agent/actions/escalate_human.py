"""Human-escalation flag. No ticket system in this hackathon build."""

from __future__ import annotations

from typing import Any

ESCALATION_FLAGS: list[dict[str, Any]] = []


def escalate_human(transaction: dict[str, Any]) -> dict[str, Any]:
    flag = {
        "escalated": True,
        "transaction_id": transaction.get("transaction_id"),
        "failure_code": transaction.get("failure_code"),
        "amount": float(transaction.get("amount") or 0),
        "reason": "Queued for human review; auto-retry disabled for this payment.",
    }
    ESCALATION_FLAGS.append(flag)
    print(
        f"[escalate] transaction_id={flag['transaction_id']} "
        f"failure_code={flag['failure_code']} amount={flag['amount']:.2f}"
    )
    return {
        "status": "escalated",
        "detail": flag["reason"],
        "amount_recovered": 0.0,
        "escalated": True,
    }
