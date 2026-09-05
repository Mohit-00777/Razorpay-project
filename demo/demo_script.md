# 3-Minute Live Demo Script
## Revenue Recovery Agent — Phase 4

> **Audience:** Judges / evaluators  
> **Format:** 3 minutes, one presenter, three terminals, one browser tab

---

## Pre-demo setup (do this 10 minutes before)

Open **four** things before you walk on stage:

| # | What | Command |
|---|------|---------|
| Terminal A | Train + run the batch | *(already done — skip if audit_log.jsonl exists)* |
| Terminal B | Dashboard server | `python dashboard/app.py` |
| Terminal C | Seed script (ready to fire) | *(open, cursor at the command)* |
| Terminal D | Report generator (ready to fire) | *(open, cursor at the command)* |
| Browser | Live dashboard | `http://127.0.0.1:8000` |

**Cold-start commands** (run from repo root with venv active):

```powershell
# First time only — model training + batch run
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python model/train_classifier.py
python agent/orchestrator.py   # populates audit/audit_log.jsonl with 60 rows

# Then start the dashboard (keep this terminal open for the whole demo)
python dashboard/app.py
```

Open `http://127.0.0.1:8000` — you should see **60 replay cards** load immediately with the batch history. The green `●  live · tailing audit_log.jsonl` pill confirms the WebSocket is tailing the log file.

---

## 0:00 – 0:25 · Idle dashboard

**Say:**

> "This is the live ops view for a bounded payment-failure recovery agent.
> Every card you see is a failed transaction that has already been processed
> from the holdout batch — classify, apply policy, take at most one action,
> write an audit line.
>
> The four counters at the top show: transactions processed, rupees at risk,
> rupees recovered, and recovery rate. The green dot means this page is
> literally tailing `audit/audit_log.jsonl` in real time — anything the agent
> writes will appear here within 200 milliseconds.
>
> This is not a chatbot. The model *proposes*; policy *decides*; the log is
> the product."

**Do:**

- Point at the `●  live` pill in the top-right — call out it's a real file tail, not polling.
- Scroll briefly to show recovered cards (green border) and escalated cards (red border).
- Do **not** click anything yet.

---

## 0:25 – 1:20 · Recoverable case (bank timeout → ₹ captured)

**Do** — switch to Terminal C and run:

```powershell
python demo/seed_failure_case.py recover
```

**Watch the browser** — a new card slides in at the bottom. Five steps reveal
one at a time (~480ms each):

| Step | Label | What the audience reads |
|------|-------|-------------------------|
| 1 | **Transaction** | `f758dce2…` failed (`bank_timeout`) |
| 2 | **Classifier** | `TEMPORARY_BANK_ISSUE` — 99% confidence |
| 3 | **Policy** | retry allowed (2/3 attempts) |
| 4 | **Action** | `retry_payment` |
| 5 | **Result** | **recovered ₹4,077** |

**Say, narrating each step as it appears:**

1. *"Transaction failed — `bank_timeout`. The gateway has already told us this
   is a transient issue, not a permanently closed account."*
2. *"Classifier says `TEMPORARY_BANK_ISSUE`, ninety-nine percent confidence.
   The model is explaining *why*, not firing the bank by itself."*
3. *"Policy checks: not a `risk_decline`, retry count is under the hard cap of
   three, backoff has elapsed — retry is allowed."*
4. *"Action: `retry_payment`. One shot, same instrument."*
5. *"Result: recovered ₹4,077. That's the original ticket amount captured on
   the retry — not a hallucinated uplift."*

**Do not** scroll or click. Let all five steps land.

---

## 1:20 – 2:10 · Unrecoverable case (permanent failure → graceful stop)

**Do** — switch to Terminal C and run:

```powershell
python demo/seed_failure_case.py escalate
```

**Watch:** A red-bordered card slides in.

| Step | Label | What the audience reads |
|------|-------|-------------------------|
| 1 | **Transaction** | `70944bdb…` failed (`invalid_vpa`) |
| 2 | **Classifier** | `PERMANENT_FAILURE` — 90% confidence |
| 3 | **Policy** | no retry — permanent failure, loop stopped |
| 4 | **Action** | `no_action` |
| 5 | **Result** | unrecoverable — marked closed, no further retries |

**Say:**

> "Second payment is `invalid_vpa` — the VPA simply doesn't exist.
> Classifier calls it `PERMANENT_FAILURE`, ninety percent confidence.
> Policy does **not** retry. Action is `no_action`. Result: marked unrecoverable
> and the loop is stopped.
>
> This is the whole point — a permanent failure is closed out of automation
> *instead* of being hammered until the retry budget is gone."

**If a judge asks about fraud:**

> "`risk_decline` is a separate **safety override** in the policy layer —
> always `ESCALATE_HUMAN`, even if the model says it's recoverable.
> No model output can override that path."

---

## 2:10 – 2:45 · Final numbers

**Do** — switch to Terminal D and run:

```powershell
python audit/report_generator.py
```

**Say the printed lines** — do not read the JSON file, just the terminal:

> "Sixty transactions processed. [read at-risk total] rupees at risk.
> [read recovered total] rupees recovered. [read recovery rate]% recovery rate.
>
> Action breakdown: mostly nudges for customer-funds issues, twenty-one
> direct retries, four escalations for risk-declines.
>
> Wasted retry cost: that's INR 5 per gateway call on retries that never
> captured — the exact figure the agent minimises by not looping.
>
> False-positive cost from Phase 2: [read misclassified count] holdout rows
> misclassified, [read INR] rupees of exposure — including recoverable money
> the model would have written off as permanent if we hadn't evaluated this."

---

## 2:45 – 3:00 · Close

**Say:**

> "Three guarantees: it is **bounded** — three retries max, no `risk_decline`
> auto-retry, permanent failures do not loop. It is **explainable** — root
> cause, confidence, policy reason, and action on every card and every audit
> line. It is **audited** — append-only JSONL and this report; `final_report.json`
> is on disk right now.
>
> The model proposes. Policy decides. The log is the product."

---

## Exact command sequence for the demo (cheat-sheet)

```powershell
# Terminal B — keep running
python dashboard/app.py

# Terminal C — fire during demo
python demo/seed_failure_case.py recover
python demo/seed_failure_case.py escalate

# Terminal D — fire at 2:10
python audit/report_generator.py
```

Browser: `http://127.0.0.1:8000`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not reach http://127.0.0.1:8000` | Start Terminal B first |
| `transaction_id not found in holdout set` | Wrong ID or holdout CSV missing — check `data/raw/holdout_set.csv` |
| Dashboard shows `disconnected · retrying` | Restart Terminal B; browser auto-reconnects |
| `No audit log` error in report_generator | Run `python agent/orchestrator.py` first |
| Steps don't animate | Hard-refresh the browser (`Ctrl+Shift+R`) |
