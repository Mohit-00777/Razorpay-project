"""Push hardcoded demo transactions through POST /trigger for a live 3-min demo.

Hardcoded IDs (do NOT change during the talk — these are confirmed in holdout_set.csv):

  recover  → f758dce2-0556-4aea-a7b0-32831d7c0098
             true_label: TEMPORARY_BANK_ISSUE | failure_code: bank_timeout | amount: ₹4,077
             Expected: classifier TEMPORARY_BANK_ISSUE (99% conf) → retry_payment → ₹4,077 recovered

  escalate → 70944bdb-26a5-44e3-90ee-7a92a09151e0
             true_label: PERMANENT_FAILURE | failure_code: invalid_vpa | amount: ₹1,136
             Expected: classifier PERMANENT_FAILURE → NO_ACTION_UNRECOVERABLE → loop stopped

Usage
-----
  python demo/seed_failure_case.py              # print IDs and exit
  python demo/seed_failure_case.py recover      # fire the recoverable case
  python demo/seed_failure_case.py escalate     # fire the permanent-failure case
  python demo/seed_failure_case.py both         # fire recover, pause, fire escalate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ── Hardcoded demo transaction IDs ────────────────────────────────────────────
# Confirmed present in data/raw/holdout_set.csv — do not change for the talk.

# CASE 1: TEMPORARY_BANK_ISSUE — bank_timeout — ₹4,077.11
# The classifier calls it recoverable and the retry mock captures the amount.
RECOVER_TRANSACTION_ID = "f758dce2-0556-4aea-a7b0-32831d7c0098"

# CASE 2: PERMANENT_FAILURE — invalid_vpa — ₹1,135.58
# The classifier calls it permanent; policy fires NO_ACTION_UNRECOVERABLE; no retry loop.
ESCALATE_TRANSACTION_ID = "70944bdb-26a5-44e3-90ee-7a92a09151e0"

CASES: dict[str, dict[str, str]] = {
    "recover": {
        "transaction_id": RECOVER_TRANSACTION_ID,
        "narrative": (
            "TEMPORARY_BANK_ISSUE - bank_timeout - INR 4,077\n"
            "  Expect: classifier TEMPORARY_BANK_ISSUE (~99% conf) ->"
            " retry_payment -> recovered INR 4,077"
        ),
    },
    "escalate": {
        "transaction_id": ESCALATE_TRANSACTION_ID,
        "narrative": (
            "PERMANENT_FAILURE - invalid_vpa - INR 1,136\n"
            "  Expect: classifier PERMANENT_FAILURE ->"
            " NO_ACTION_UNRECOVERABLE -> loop stopped, INR 0 recovered"
        ),
    },
}

DEFAULT_URL = "http://127.0.0.1:8000/trigger"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def trigger(transaction_id: str, url: str) -> dict:
    """POST {transaction_id} to /trigger and return the parsed JSON response."""
    payload = json.dumps({"transaction_id": transaction_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"\n[FAIL] HTTP {exc.code} from {url}\n  {detail}\n"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"\n[FAIL] Could not reach {url}\n"
            f"  Is the dashboard running?  ->  python dashboard/app.py\n"
            f"  {exc}\n"
        ) from exc


def _format_record(record: dict) -> str:
    conf     = float(record.get("classifier_confidence") or 0)
    cause    = record.get("predicted_root_cause", "?")
    action   = record.get("final_action_taken", "?")
    rec_amt  = float(record.get("amount_recovered") or 0)
    at_risk  = float(record.get("amount") or 0)
    pol      = (record.get("policy_reason") or "")[:120]
    return (
        f"  predicted_root_cause : {cause}  ({conf:.0%} confidence)\n"
        f"  final_action_taken   : {action}\n"
        f"  amount at risk       : INR {at_risk:,.2f}\n"
        f"  amount_recovered     : INR {rec_amt:,.2f}\n"
        f"  policy_reason        : {pol}"
    )


# ── Case runner ────────────────────────────────────────────────────────────────

def run_case(name: str, url: str) -> dict:
    case = CASES[name]
    tid  = case["transaction_id"]

    print(f"\n{'-'*56}")
    print(f"  DEMO CASE: {name.upper()}")
    print(f"{'-'*56}")
    print(case["narrative"])
    print(f"\n  POST {url}")
    print(f"  transaction_id = {tid}")
    print()

    record = trigger(tid, url)
    print("  [OK] Response:")
    print(_format_record(record))
    print(f"{'-'*56}")
    print("  -> Dashboard card should appear within ~200 ms")
    return record


# ── CLI entry-point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seed_failure_case.py",
        description="Fire hardcoded demo transactions through the live dashboard.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "case",
        nargs="?",
        choices=["recover", "escalate", "both"],
        default=None,
        help=(
            "Which demo case to fire:\n"
            "  recover   — TEMPORARY_BANK_ISSUE → retry_payment → ₹ captured\n"
            "  escalate  — PERMANENT_FAILURE → no_action → loop stopped\n"
            "  both      — fire recover, wait, fire escalate\n"
            "(omit to print IDs and exit)"
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Dashboard /trigger URL  (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help="Seconds to pause between 'both' cases so the audience can read the card  (default: 8)",
    )
    args = parser.parse_args()

    if args.case is None:
        print("\nHardcoded demo transaction IDs:\n")
        print(f"  recover   {RECOVER_TRANSACTION_ID}")
        print(f"  escalate  {ESCALATE_TRANSACTION_ID}")
        print(
            "\nUsage:\n"
            "  python demo/seed_failure_case.py recover\n"
            "  python demo/seed_failure_case.py escalate\n"
            "  python demo/seed_failure_case.py both\n"
        )
        return

    if args.case == "both":
        run_case("recover",  args.url)
        print(f"\n  Pausing {args.delay:.0f}s -- let the audience read the card...\n")
        time.sleep(args.delay)
        run_case("escalate", args.url)
        print("\n  Both cases fired. Switch to the report terminal when ready.\n")
        return

    run_case(args.case, args.url)


if __name__ == "__main__":
    main()
