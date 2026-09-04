"""Judge-facing map: predicted root cause -> default recovery action.

This is a policy table, not a second model. Defaults match the Phase 1
`ideal_action` vocabulary. Amount and retry_count can still refine the
action later; these are the explainable first recommendations.
"""

from __future__ import annotations

# Explicit 1:1 policy. Keys are true_root_cause_category values.
INTERVENTION_MAP: dict[str, dict[str, str]] = {
    "TEMPORARY_BANK_ISSUE": {
        "action": "RETRY_IMMEDIATE",
        "justification": (
            "Timeouts and network errors are usually short-lived at the bank or "
            "switch. Retry the same instrument immediately (or after a brief "
            "backoff if retries are already high) instead of waiting a day or "
            "writing the payment off."
        ),
    },
    "CUSTOMER_FUNDS_ISSUE": {
        "action": "SEND_NUDGE",
        "justification": (
            "Insufficient funds and many generic card declines are a cash-flow "
            "problem, not a broken rail. Nudge the customer (and if they already "
            "failed once, schedule RETRY_DELAYED_24H / 72H around typical salary "
            "cycles) rather than hammering the bank immediately."
        ),
    },
    "MANDATE_EXPIRED": {
        "action": "SEND_NUDGE",
        "justification": (
            "An expired e-mandate cannot be recovered by retrying the same debit. "
            "Ask the customer to re-authorize; escalate to a human only for large "
            "tickets that need a new mandate papered."
        ),
    },
    "RISK_BLOCKED": {
        "action": "ESCALATE_HUMAN",
        "justification": (
            "Risk / fraud declines should not be auto-retried. A human review "
            "avoids retrying a blocked instrument and creating more declines."
        ),
    },
    "PERMANENT_FAILURE": {
        "action": "NO_ACTION_UNRECOVERABLE",
        "justification": (
            "Invalid VPA, closed/stolen cards, and similar hard failures will not "
            "succeed on retry. Stop automated recovery so we do not spend retries "
            "or annoy the customer; recover only if they update the instrument."
        ),
    },
}


def recommend_action(root_cause_category: str) -> str:
    try:
        return INTERVENTION_MAP[root_cause_category]["action"]
    except KeyError as exc:
        raise KeyError(f"No intervention mapped for {root_cause_category!r}") from exc
