"""Hard-coded, bounded recovery policies. Independent of the classifier."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make model/ importable so we can reference the shared threshold constant.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(_ROOT / "model"))

from predict import CONFIDENCE_THRESHOLD  # noqa: E402

MAX_RETRY_ATTEMPTS = 3
BACKOFF_HOURS = (2, 24, 72)  # after 0, 1, 2 prior retries
RETRY_ACTIONS = {
    "RETRY_IMMEDIATE",
    "RETRY_DELAYED_24H",
    "RETRY_DELAYED_72H",
}
RISK_FAILURE_CODE = "risk_decline"


@dataclass(frozen=True)
class PolicyDecision:
    final_action: str
    policy_reason: str
    override_fired: bool
    low_confidence_override_fired: bool
    retry_attempt_number: int
    backoff_hours_required: int | None
    backoff_elapsed: bool
    execute_retry: bool


def _parse_timestamp(raw: Any) -> datetime | None:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_failure(transaction: dict[str, Any], now: datetime | None = None) -> float | None:
    ts = _parse_timestamp(transaction.get("failure_timestamp"))
    if ts is None:
        return None
    clock = now or datetime.now(timezone.utc)
    return max(0.0, (clock - ts).total_seconds() / 3600.0)


def decide(
    transaction: dict[str, Any],
    model_recommended_action: str,
    now: datetime | None = None,
) -> PolicyDecision:
    """Return the policy-enforced action plus a plain-English reason.

    Override priority (highest → lowest):
      1. risk_decline hard-block  → override_fired=True
      2. low model confidence     → low_confidence_override_fired=True
      3. retry-cap escalation     → override_fired=True
      4. normal path              → both False
    """
    failure_code = str(transaction.get("failure_code") or "")
    retry_so_far = int(transaction.get("retry_count_so_far") or 0)
    confidence = float(transaction.get("classifier_confidence") or 1.0)
    recommended = str(model_recommended_action)

    # ── Rule 1: risk_decline hard-block (highest priority) ────────────────────
    if failure_code == RISK_FAILURE_CODE:
        return PolicyDecision(
            final_action="ESCALATE_HUMAN",
            policy_reason=(
                "SAFETY OVERRIDE fired: raw failure_code is risk_decline, so this "
                "payment is forced to ESCALATE_HUMAN and must not be auto-retried, "
                f"even though the classifier recommended {recommended}."
            ),
            override_fired=True,
            low_confidence_override_fired=False,
            retry_attempt_number=retry_so_far,
            backoff_hours_required=None,
            backoff_elapsed=True,
            execute_retry=False,
        )

    # ── Rule 2: low model confidence gate ─────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return PolicyDecision(
            final_action="ESCALATE_HUMAN",
            policy_reason=(
                f"LOW CONFIDENCE OVERRIDE fired: classifier top-class probability "
                f"{confidence:.4f} is below threshold {CONFIDENCE_THRESHOLD}. "
                f"Routing to human review instead of following the model's "
                f"recommendation ({recommended}). Increase model training data or "
                f"lower CONFIDENCE_THRESHOLD to reduce human escalation rate."
            ),
            override_fired=False,
            low_confidence_override_fired=True,
            retry_attempt_number=retry_so_far,
            backoff_hours_required=None,
            backoff_elapsed=True,
            execute_retry=False,
        )

    # ── Rule 3: retry-cap escalation ─────────────────────────────────────────
    if recommended in RETRY_ACTIONS and retry_so_far >= MAX_RETRY_ATTEMPTS:
        return PolicyDecision(
            final_action="ESCALATE_HUMAN",
            policy_reason=(
                f"Retry cap reached: this transaction already has {retry_so_far} "
                f"attempts (max {MAX_RETRY_ATTEMPTS}). Forcing ESCALATE_HUMAN; "
                f"no further retries. Classifier had recommended {recommended}."
            ),
            override_fired=True,
            low_confidence_override_fired=False,
            retry_attempt_number=retry_so_far,
            backoff_hours_required=None,
            backoff_elapsed=True,
            execute_retry=False,
        )

    # ── Rule 4: normal path ───────────────────────────────────────────────────
    if recommended not in RETRY_ACTIONS:
        used_model = recommended == model_recommended_action
        return PolicyDecision(
            final_action=recommended,
            policy_reason=(
                "No safety override. Using the classifier's recommendation "
                f"{recommended} because this is not a risk_decline and the retry "
                f"cap ({retry_so_far}/{MAX_RETRY_ATTEMPTS}) has not been hit."
                if used_model
                else f"Using policy action {recommended}."
            ),
            override_fired=False,
            low_confidence_override_fired=False,
            retry_attempt_number=retry_so_far,
            backoff_hours_required=None,
            backoff_elapsed=True,
            execute_retry=False,
        )

    backoff = BACKOFF_HOURS[min(retry_so_far, len(BACKOFF_HOURS) - 1)]
    elapsed_hours = hours_since_failure(transaction, now=now)
    backoff_elapsed = elapsed_hours is None or elapsed_hours >= backoff
    next_attempt = retry_so_far + 1

    if not backoff_elapsed:
        return PolicyDecision(
            final_action=recommended,
            policy_reason=(
                f"Classifier recommended {recommended}. Cap allows attempt "
                f"{next_attempt}/{MAX_RETRY_ATTEMPTS}, but the minimum backoff is "
                f"{backoff} hours since the last failure and only "
                f"{elapsed_hours:.1f} hours have passed. Deferring the retry; "
                "not calling the payment API this cycle."
            ),
            override_fired=False,
            low_confidence_override_fired=False,
            retry_attempt_number=retry_so_far,
            backoff_hours_required=backoff,
            backoff_elapsed=False,
            execute_retry=False,
        )

    return PolicyDecision(
        final_action=recommended,
        policy_reason=(
            f"No safety override. Using the classifier's recommendation "
            f"{recommended}. Executing retry attempt {next_attempt}/"
            f"{MAX_RETRY_ATTEMPTS} after the {backoff}h backoff elapsed "
            f"({elapsed_hours:.1f}h since last failure)."
            if elapsed_hours is not None
            else (
                f"No safety override. Using the classifier's recommendation "
                f"{recommended}. Executing retry attempt {next_attempt}/"
                f"{MAX_RETRY_ATTEMPTS} (no failure_timestamp; backoff skipped)."
            )
        ),
        override_fired=False,
        low_confidence_override_fired=False,
        retry_attempt_number=next_attempt,
        backoff_hours_required=backoff,
        backoff_elapsed=True,
        execute_retry=True,
    )

