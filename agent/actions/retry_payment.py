"""Retry a one-shot payment via the (mocked) Razorpay client."""

from __future__ import annotations

from typing import Any

from razorpay_client import default_client


def retry_payment(transaction: dict[str, Any], client=default_client) -> dict[str, Any]:
    amount_inr = float(transaction.get("amount") or 0)
    paise = int(round(amount_inr * 100))
    order = client.order.create(
        {
            "amount": paise,
            "currency": "INR",
            "receipt": str(transaction.get("transaction_id")),
        }
    )
    # UPI-style one-shot: create returns captured or failed (same shape as SDK).
    payment = client.payment.create(
        {
            "amount": paise,
            "currency": "INR",
            "order_id": order["id"],
            "receipt": str(transaction.get("transaction_id")),
            "failure_code": transaction.get("failure_code"),
            "notes": {"failure_code": transaction.get("failure_code")},
        }
    )
    recovered = amount_inr if payment.get("status") == "captured" else 0.0
    if recovered:
        return {
            "status": "captured",
            "detail": f"Mock payment {payment.get('id')} captured",
            "amount_recovered": recovered,
            "razorpay_order": order,
            "razorpay_payment": payment,
        }
    return {
        "status": "failed",
        "detail": (
            f"Mock payment {payment.get('id')} failed "
            f"({payment.get('error_code')}: {payment.get('error_description')})"
        ),
        "amount_recovered": 0.0,
        "razorpay_order": order,
        "razorpay_payment": payment,
    }
