# Revenue Recovery Agent

Hackathon project for **Razorpay Buildathon — Revenue Recovery**.

The agent maps **payment degradation → root cause → recovery action** so failed or degraded transactions can be retried, nudged, or escalated instead of leaking revenue.

This repo is synthetic-data first. There is no Razorpay SDK and no live API keys: PAN/KYC checks are out of scope for the hackathon timeline.

## Phase 1 (current)

Generate a labeled batch of 300 failed/degraded payments for classifier training.

```text
revenue-recovery-agent/
├── README.md
├── requirements.txt
└── data/
    ├── generate_synthetic_failures.py
    ├── raw/failures_batch.csv      # created by the generator
    └── schema.md
```

### Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate data

```bash
python data/generate_synthetic_failures.py
```

Writes `data/raw/failures_batch.csv` and prints `true_root_cause_category` value counts.

Field definitions, canonical `failure_code` → label maps, and the ~10% label-noise rule are in [`data/schema.md`](data/schema.md).

## Later phases (not built yet)

- Root-cause classifier (`TEMPORARY_BANK_ISSUE`, `CUSTOMER_FUNDS_ISSUE`, `MANDATE_EXPIRED`, `RISK_BLOCKED`, `PERMANENT_FAILURE`)
- Recovery-action policy (`RETRY_*`, `SEND_NUDGE`, `ESCALATE_HUMAN`, `NO_ACTION_UNRECOVERABLE`)
- FastAPI serving layer
