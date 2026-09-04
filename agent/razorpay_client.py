"""Mock Razorpay client (same call shapes as the official Python SDK).

This module stands in for real Razorpay *test-mode* API calls. Activating a
test-mode account requires PAN/KYC that we cannot complete on the hackathon
timeline, so there are no API keys, no `razorpay` package, and no network I/O.

To swap in production later, replace this file with a thin wrapper around
`razorpay.Client(auth=(key_id, key_secret))`. Call sites already use
`client.payment.capture(...)` / `client.payment.create(...)` / `client.order.create(...)`
and expect dicts with `id`, `status`, `error_code`, `error_description`,
`amount`, `created_at`.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

# Paise, matching the real SDK.
CURRENCY = "INR"

# Mock success rates by the *original* gateway failure_code (simulation only;
# the real API does not take this field). network_error is intentionally noisy
# (~40%) because the holdout classifier often cannot tell transient vs permanent.
SUCCESS_RATE_BY_FAILURE_CODE: dict[str, float] = {
    "bank_timeout": 0.70,
    "network_error": 0.40,
    "risk_decline": 0.10,
    "insufficient_funds": 0.25,
    "card_declined": 0.20,
    "expired_mandate": 0.0,  # never, unless a fresh mandate token is supplied
    "invalid_vpa": 0.0,
}


def _stable_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _mock_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}{digest}"


def _now() -> int:
    return int(time.time())


def _simulate_payment(data: dict[str, Any], kind: str) -> dict[str, Any]:
    amount = int(data.get("amount") or 0)
    receipt = str(data.get("receipt") or data.get("payment_id") or "unknown")
    failure_code = str(data.get("failure_code") or data.get("notes", {}).get("failure_code") or "")
    fresh_mandate = bool(data.get("fresh_mandate") or data.get("token"))

    rate = SUCCESS_RATE_BY_FAILURE_CODE.get(failure_code, 0.30)
    if failure_code == "expired_mandate" and fresh_mandate:
        rate = 0.80
    elif failure_code == "expired_mandate":
        rate = 0.0

    pay_id = _mock_id("pay_mock_", kind, receipt, str(amount))
    created_at = _now()
    roll = _stable_unit_interval(kind, receipt, failure_code, str(amount))

    if roll < rate:
        return {
            "id": pay_id,
            "entity": "payment",
            "amount": amount,
            "currency": CURRENCY,
            "status": "captured",
            "captured": True,
            "error_code": None,
            "error_description": None,
            "created_at": created_at,
        }

    error_code = failure_code or "payment_failed"
    descriptions = {
        "bank_timeout": "The bank did not respond in time.",
        "network_error": "Network error while contacting the acquirer.",
        "risk_decline": "Payment declined by risk engine.",
        "expired_mandate": "Mandate is expired; a new mandate is required.",
        "invalid_vpa": "Invalid VPA.",
        "insufficient_funds": "Insufficient funds.",
        "card_declined": "Card declined by issuer.",
    }
    return {
        "id": pay_id,
        "entity": "payment",
        "amount": amount,
        "currency": CURRENCY,
        "status": "failed",
        "captured": False,
        "error_code": error_code,
        "error_description": descriptions.get(error_code, "Payment failed."),
        "created_at": created_at,
    }


class Payment:
    """Mirrors `razorpay.resources.payment.Payment`."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def capture(self, payment_id: str, amount: int, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(data or {})
        payload["payment_id"] = payment_id
        payload["amount"] = int(amount)
        payload.setdefault("receipt", payment_id)
        return _simulate_payment(payload, kind="capture")

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        return _simulate_payment(dict(data), kind="create")

    def fetch(self, payment_id: str) -> dict[str, Any]:
        return {
            "id": payment_id,
            "entity": "payment",
            "status": "created",
            "amount": 0,
            "currency": CURRENCY,
            "error_code": None,
            "error_description": None,
            "created_at": _now(),
        }


class Order:
    """Mirrors `razorpay.resources.order.Order`."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        receipt = str(data.get("receipt") or "order")
        amount = int(data.get("amount") or 0)
        return {
            "id": _mock_id("order_mock_", receipt, str(amount)),
            "entity": "order",
            "amount": amount,
            "currency": data.get("currency", CURRENCY),
            "status": "created",
            "receipt": receipt,
            "created_at": _now(),
        }


class Token:
    """Mirrors token/mandate APIs used for e-mandates."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("fresh_mandate"):
            return {
                "id": None,
                "entity": "token",
                "status": "failed",
                "error_code": "expired_mandate",
                "error_description": "Cannot retry an expired mandate without a fresh mandate.",
                "amount": int(data.get("amount") or 0),
                "created_at": _now(),
            }
        customer_id = str(data.get("customer_id") or "cust")
        return {
            "id": _mock_id("token_mock_", customer_id),
            "entity": "token",
            "status": "confirmed",
            "error_code": None,
            "error_description": None,
            "amount": int(data.get("amount") or 0),
            "created_at": _now(),
        }


class Client:
    """Drop-in mock for `razorpay.Client`."""

    def __init__(self, auth: tuple[str, str] | None = None, **kwargs: Any) -> None:
        self.auth = auth
        self.payment = Payment(self)
        self.payments = self.payment
        self.order = Order(self)
        self.orders = self.order
        self.token = Token(self)


default_client = Client(auth=("mock_key_id", "mock_key_secret"))
