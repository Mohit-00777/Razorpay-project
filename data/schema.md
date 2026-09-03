# Synthetic failure batch schema

Source script: `data/generate_synthetic_failures.py`  
Output: `data/raw/failures_batch.csv` (300 rows, UTF-8, header row)

Seed is fixed (`RANDOM_SEED = 42`) so reruns are reproducible.

## Columns

| Field | Type | Description |
| --- | --- | --- |
| `transaction_id` | UUID string | Unique payment attempt id. |
| `customer_id` | string | Synthetic payer id (`cust_` + 8 hex chars). Drawn from a pool of 80 customers so some payers appear more than once. |
| `amount` | float (INR) | Ticket size in rupees, 2 decimal places. Log-normal around typical checkout values, clipped to **[100, 50000]**. UPI rows are biased slightly smaller; cards/netbanking/e-mandate slightly larger. |
| `payment_method` | enum | One of `upi`, `card`, `netbanking`, `emandate`. Constrained by `failure_code` (see below). |
| `failure_code` | enum | Gateway-style reason: `insufficient_funds`, `bank_timeout`, `expired_mandate`, `risk_decline`, `card_declined`, `network_error`, `invalid_vpa`. |
| `failure_timestamp` | ISO-8601 UTC | Failure time in the last 14 days (timezone `Z`). |
| `retry_count_so_far` | int | Prior auto-retries already attempted, **0–3**. Temporary bank/network failures are more likely to already have a retry. |
| `bank_name` | string | Issuing / destination bank (Indian retail set used by the generator). |
| `true_root_cause_category` | enum | Ground-truth **root cause** label (with ~10% noise; see Label logic). |
| `ideal_action` | enum | Ground-truth **recovery** label, derived from the *canonical* (pre-noise) root cause plus `retry_count_so_far` and `amount`. |

### Enums

**Root cause**

- `TEMPORARY_BANK_ISSUE`
- `CUSTOMER_FUNDS_ISSUE`
- `MANDATE_EXPIRED`
- `RISK_BLOCKED`
- `PERMANENT_FAILURE`

**Ideal action**

- `RETRY_IMMEDIATE`
- `RETRY_DELAYED_24H`
- `RETRY_DELAYED_72H`
- `SEND_NUDGE`
- `ESCALATE_HUMAN`
- `NO_ACTION_UNRECOVERABLE`

## Method ↔ failure_code constraints

These combinations are never generated:

- `invalid_vpa` only with `upi`
- `expired_mandate` only with `emandate`
- `card_declined` only with `card`

All other codes may appear on any method (weighted toward UPI).

## Label logic

### 1. Canonical `failure_code` → root cause

Used for almost every row before noise is applied.

| `failure_code` | Canonical `true_root_cause_category` |
| --- | --- |
| `bank_timeout` | `TEMPORARY_BANK_ISSUE` |
| `network_error` | `TEMPORARY_BANK_ISSUE` |
| `insufficient_funds` | `CUSTOMER_FUNDS_ISSUE` |
| `expired_mandate` | `MANDATE_EXPIRED` (never noised) |
| `risk_decline` | `RISK_BLOCKED` |
| `invalid_vpa` | `PERMANENT_FAILURE` |
| `card_declined` | Stochastic but realistic: **70%** `CUSTOMER_FUNDS_ISSUE` (generic issuer decline / limit), **20%** `RISK_BLOCKED` (fraud / velocity), **10%** `PERMANENT_FAILURE` (stolen / closed / do-not-honor) |

`bank_timeout` is therefore *usually* `TEMPORARY_BANK_ISSUE`; `expired_mandate` is *always* `MANDATE_EXPIRED`.

### 2. Canonical root cause + context → `ideal_action`

`ideal_action` is computed from the **canonical** root cause (step 1), not from the possibly noised label. That keeps the recovery target operationally correct even when the cause label is flipped.

| Canonical cause | Rule |
| --- | --- |
| `TEMPORARY_BANK_ISSUE` | `retry_count_so_far` 0–1 → `RETRY_IMMEDIATE`; 2 → `RETRY_DELAYED_24H`; 3 → `ESCALATE_HUMAN` |
| `CUSTOMER_FUNDS_ISSUE` | First seen (`retry_count_so_far == 0`) → `SEND_NUDGE`; else if `amount < 5000` → `RETRY_DELAYED_24H`; else → `RETRY_DELAYED_72H` (salary / cash-flow cycle) |
| `MANDATE_EXPIRED` | `amount >= 10000` → `ESCALATE_HUMAN` (re-paper mandate); else → `SEND_NUDGE` |
| `RISK_BLOCKED` | Always `ESCALATE_HUMAN` |
| `PERMANENT_FAILURE` | Always `NO_ACTION_UNRECOVERABLE` |

### 3. Label noise (~10%)

After steps 1–2, each row with `failure_code != expired_mandate` is flipped with probability **0.10**: `true_root_cause_category` is replaced by a uniformly chosen *different* root-cause class.

- Intended effect: `failure_code` → cause is consistent enough to learn, but not a trivial 1:1 lookup.
- `expired_mandate` rows are **excluded** from noise so they stay `MANDATE_EXPIRED`.
- `ideal_action` is **not** flipped, so ~10% of rows have a cause label that disagrees with the recovery label (messy production analog).

## Class mix (generator sampling, before noise)

Failure codes are sampled with these approximate weights so all five cause classes appear:

- `insufficient_funds` 0.22, `bank_timeout` 0.22, `card_declined` 0.14, `network_error` 0.12, `expired_mandate` 0.12, `risk_decline` 0.10, `invalid_vpa` 0.08

Exact post-noise counts are printed by the script (`value_counts` of `true_root_cause_category`).
