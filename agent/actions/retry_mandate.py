"""Retry an e-mandate debit via the (mocked) Razorpay client."""

from __future__ import annotations

from typing import Any

from razorpay_client import default_client


def retry_mandate(transaction: dict[str, Any], client=default_client) -> dict[str, Any]:
    amount_inr = float(transaction.get("amount") or 0)
    paise = int(round(amount_inr * 100))
    # No fresh mandate on file in this dataset — token.create must fail.
    token = client.token.create(
        {
            "customer_id": transaction.get("customer_id"),
            "amount": paise,
            "fresh_mandate": False,
        }
    )
    if token.get("status") != "confirmed" or not token.get("id"):
        return {
            "status": "failed",
            "detail": (
                "Expired mandate cannot be retried without a new customer "
                f"authorization ({token.get('error_description')})"
            ),
            "amount_recovered": 0.0,
            "razorpay_token": token,
        }

    payment = client.payment.create(
        {
            "amount": paise,
            "currency": "INR",
            "receipt": str(transaction.get("transaction_id")),
            "token": token["id"],
            "fresh_mandate": True,
            "failure_code": transaction.get("failure_code"),
            "notes": {"failure_code": transaction.get("failure_code")},
        }
    )
    recovered = amount_inr if payment.get("status") == "captured" else 0.0
    return {
        "status": payment.get("status") or "failed",
        "detail": f"Mandate debit {payment.get('id')} status={payment.get('status')}",
        "amount_recovered": recovered,
        "razorpay_payment": payment,
        "razorpay_token": token,
    }
