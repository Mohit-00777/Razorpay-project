"""Generate a labeled synthetic batch of failed / degraded payments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

RANDOM_SEED = 42
N_ROWS = 300
N_CUSTOMERS = 80
NOISE_RATE = 0.10

OUT_PATH = Path(__file__).resolve().parent / "raw" / "failures_batch.csv"

FAILURE_CODES = [
    "insufficient_funds",
    "bank_timeout",
    "expired_mandate",
    "risk_decline",
    "card_declined",
    "network_error",
    "invalid_vpa",
]
FAILURE_WEIGHTS = np.array([0.22, 0.22, 0.12, 0.10, 0.14, 0.12, 0.08])

ROOT_CAUSES = [
    "TEMPORARY_BANK_ISSUE",
    "CUSTOMER_FUNDS_ISSUE",
    "MANDATE_EXPIRED",
    "RISK_BLOCKED",
    "PERMANENT_FAILURE",
]

BANKS = [
    "HDFC Bank",
    "ICICI Bank",
    "State Bank of India",
    "Axis Bank",
    "Kotak Mahindra Bank",
    "Yes Bank",
    "IDFC FIRST Bank",
    "Punjab National Bank",
    "Bank of Baroda",
    "Union Bank of India",
    "IndusInd Bank",
    "Federal Bank",
]

METHOD_WEIGHTS = {
    "upi": 0.55,
    "card": 0.25,
    "netbanking": 0.12,
    "emandate": 0.08,
}


def canonical_root_cause(failure_code: str, rng: np.random.Generator) -> str:
    if failure_code in ("bank_timeout", "network_error"):
        return "TEMPORARY_BANK_ISSUE"
    if failure_code == "insufficient_funds":
        return "CUSTOMER_FUNDS_ISSUE"
    if failure_code == "expired_mandate":
        return "MANDATE_EXPIRED"
    if failure_code == "risk_decline":
        return "RISK_BLOCKED"
    if failure_code == "invalid_vpa":
        return "PERMANENT_FAILURE"
    if failure_code == "card_declined":
        return rng.choice(
            ["CUSTOMER_FUNDS_ISSUE", "RISK_BLOCKED", "PERMANENT_FAILURE"],
            p=[0.70, 0.20, 0.10],
        )
    raise ValueError(f"unknown failure_code: {failure_code}")


def payment_method_for(failure_code: str, rng: np.random.Generator) -> str:
    if failure_code == "invalid_vpa":
        return "upi"
    if failure_code == "expired_mandate":
        return "emandate"
    if failure_code == "card_declined":
        return "card"
    methods = list(METHOD_WEIGHTS.keys())
    probs = np.array(list(METHOD_WEIGHTS.values()), dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(methods, p=probs))


def sample_amount(method: str, rng: np.random.Generator) -> float:
    # Log-normal in INR, then clip. UPI skews smaller; others a bit larger.
    mu, sigma = (7.4, 0.85) if method == "upi" else (8.1, 0.9)
    raw = float(rng.lognormal(mean=mu, sigma=sigma))
    if method == "upi":
        raw *= 0.55
    amount = float(np.clip(raw, 100.0, 50_000.0))
    return round(amount, 2)


def sample_retry_count(failure_code: str, rng: np.random.Generator) -> int:
    if failure_code in ("bank_timeout", "network_error"):
        probs = [0.25, 0.30, 0.28, 0.17]
    elif failure_code in ("insufficient_funds", "card_declined"):
        probs = [0.45, 0.30, 0.18, 0.07]
    else:
        probs = [0.55, 0.25, 0.15, 0.05]
    return int(rng.choice([0, 1, 2, 3], p=probs))


def ideal_action(canonical_cause: str, retry_count: int, amount: float) -> str:
    if canonical_cause == "TEMPORARY_BANK_ISSUE":
        if retry_count <= 1:
            return "RETRY_IMMEDIATE"
        if retry_count == 2:
            return "RETRY_DELAYED_24H"
        return "ESCALATE_HUMAN"
    if canonical_cause == "CUSTOMER_FUNDS_ISSUE":
        if retry_count == 0:
            return "SEND_NUDGE"
        if amount < 5000:
            return "RETRY_DELAYED_24H"
        return "RETRY_DELAYED_72H"
    if canonical_cause == "MANDATE_EXPIRED":
        return "ESCALATE_HUMAN" if amount >= 10_000 else "SEND_NUDGE"
    if canonical_cause == "RISK_BLOCKED":
        return "ESCALATE_HUMAN"
    if canonical_cause == "PERMANENT_FAILURE":
        return "NO_ACTION_UNRECOVERABLE"
    raise ValueError(f"unknown cause: {canonical_cause}")


def apply_label_noise(
    failure_code: str, cause: str, rng: np.random.Generator
) -> str:
    if failure_code == "expired_mandate":
        return "MANDATE_EXPIRED"
    if rng.random() >= NOISE_RATE:
        return cause
    others = [c for c in ROOT_CAUSES if c != cause]
    return str(rng.choice(others))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    fake = Faker()
    Faker.seed(RANDOM_SEED)

    customers = [f"cust_{fake.hexify(text='^^^^^^^^')}" for _ in range(N_CUSTOMERS)]
    now = datetime.now(timezone.utc)
    codes = rng.choice(FAILURE_CODES, size=N_ROWS, p=FAILURE_WEIGHTS / FAILURE_WEIGHTS.sum())

    rows = []
    for failure_code in codes:
        method = payment_method_for(failure_code, rng)
        amount = sample_amount(method, rng)
        retry_count = sample_retry_count(failure_code, rng)
        canonical = canonical_root_cause(failure_code, rng)
        labeled = apply_label_noise(failure_code, canonical, rng)
        offset_s = int(rng.integers(0, 14 * 24 * 3600))
        ts = (now - timedelta(seconds=offset_s)).replace(microsecond=0)

        rows.append(
            {
                "transaction_id": str(fake.uuid4()),
                "customer_id": str(rng.choice(customers)),
                "amount": amount,
                "payment_method": method,
                "failure_code": failure_code,
                "failure_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "retry_count_so_far": retry_count,
                "bank_name": str(rng.choice(BANKS)),
                "true_root_cause_category": labeled,
                "ideal_action": ideal_action(canonical, retry_count, amount),
            }
        )

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    print("\ntrue_root_cause_category value_counts:")
    print(df["true_root_cause_category"].value_counts().to_string())
    print("\nideal_action value_counts:")
    print(df["ideal_action"].value_counts().to_string())


if __name__ == "__main__":
    main()
