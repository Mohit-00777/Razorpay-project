"""Customer nudge templates (English + Hinglish). No network send."""

from __future__ import annotations

from typing import Any

TEMPLATES = {
    "insufficient_funds": {
        "en": (
            "Your payment of INR {amount:.2f} could not go through due to insufficient "
            "balance. Please add funds and retry from the same link."
        ),
        "hinglish": (
            "Aapka INR {amount:.2f} ka payment balance kam hone ki wajah se fail ho gaya. "
            "Kripya account mein paise add karke dubara try karein."
        ),
    },
    "expired_mandate": {
        "en": (
            "Your e-mandate for INR {amount:.2f} has expired. Please re-authorize the "
            "mandate so we can collect the payment."
        ),
        "hinglish": (
            "Aapka e-mandate (INR {amount:.2f}) expire ho chuka hai. Kripya naya mandate "
            "authorize karein taaki payment complete ho sake."
        ),
    },
    "default": {
        "en": (
            "We could not complete your payment of INR {amount:.2f}. Please retry with "
            "an updated payment method if this continues."
        ),
        "hinglish": (
            "Aapka INR {amount:.2f} ka payment complete nahi ho paya. Kripya updated "
            "payment method se retry karein."
        ),
    },
}


def send_nudge(transaction: dict[str, Any]) -> dict[str, Any]:
    amount = float(transaction.get("amount") or 0)
    code = str(transaction.get("failure_code") or "default")
    pair = TEMPLATES.get(code, TEMPLATES["default"])
    message_en = pair["en"].format(amount=amount)
    message_hinglish = pair["hinglish"].format(amount=amount)
    detail = (
        f"[nudge en] {message_en} | [nudge hinglish] {message_hinglish}"
    )
    print(detail)
    return {
        "status": "nudge_logged",
        "detail": detail,
        "amount_recovered": 0.0,
        "message_en": message_en,
        "message_hinglish": message_hinglish,
    }
