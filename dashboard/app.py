import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"

# Make all project sub-packages importable from any working directory.
for _p in (
    ROOT / "agent",
    ROOT / "agent" / "actions",
    ROOT / "model",
    ROOT / "audit",
    ROOT / "dashboard",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from orchestrator import process_transaction, process_transaction_id  # noqa: E402
from websocket_feed import follow_audit_log  # noqa: E402

app = FastAPI(
    title="Revenue Recovery Agent",
    version="0.5.0",
    description=(
        "Bounded, explainable, audited payment-failure recovery. "
        "Classify → Policy → Action → Audit."
    ),
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TriggerRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1)


# Known failure codes (kept in sync with data/schema.md)
_FAILURE_CODES = [
    "insufficient_funds",
    "card_declined",
    "do_not_honor",
    "generic_decline",
    "bank_timeout",
    "gateway_timeout",
    "network_error",
    "expired_mandate",
    "risk_decline",
    "invalid_vpa",
    "closed_account",
    "stolen_card",
    "lost_card",
    "card_expired",
]

_PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]


class ManualTransactionRequest(BaseModel):
    """Payload for /manual-transaction — mirrors the holdout-set schema."""

    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    failure_code: str = Field(..., description="One of the known failure codes")
    payment_method: str = Field(..., description="card | upi | netbanking | wallet")
    retry_count_so_far: int = Field(0, ge=0, le=10)
    bank_name: str = Field("UNKNOWN_BANK", description="Issuing bank name")
    dry_run: bool = Field(
        False,
        description=(
            "When True: run classification + policy but do NOT execute any action "
            "and do NOT write to audit_log.jsonl. Returns full reasoning only."
        ),
    )


@app.get("/")
async def index() -> FileResponse:
    """Serve the live dashboard SPA."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "audit_log": str(ROOT / "audit" / "audit_log.jsonl")}


@app.get("/bank-names")
def bank_names() -> list[str]:
    """Return sorted distinct bank_name values from the synthetic dataset CSV."""
    csv_path = ROOT / "data" / "raw" / "failures_batch.csv"
    if not csv_path.exists():
        return []
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_csv(csv_path, usecols=["bank_name"])
        return sorted(df["bank_name"].dropna().unique().tolist())
    except Exception:  # noqa: BLE001
        return []


@app.post("/trigger")
def trigger(body: TriggerRequest) -> dict[str, Any]:
    """Run the orchestrator on one holdout transaction and append to the audit log.

    The newly appended audit line is immediately picked up by the WebSocket
    tail loop and pushed to every connected browser — so the dashboard card
    appears within ~200 ms of this call returning.
    """
    tid = body.transaction_id.strip()
    try:
        record = process_transaction_id(tid)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"transaction_id {tid!r} not found in holdout set. {exc}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Required file missing — did you run train_classifier.py? {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return record


@app.post("/report")
def generate_report() -> dict[str, Any]:
    """Re-generate final_report.json from the current audit log and return it."""
    try:
        from report_generator import build_report  # noqa: PLC0415

        report = build_report()
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        report_path = ROOT / "audit" / "final_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return report
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/manual-transaction")
def manual_transaction(body: ManualTransactionRequest) -> dict[str, Any]:
    """Run the full orchestrator pipeline on a manually-entered transaction.

    Live mode (dry_run=False):
        - Runs classification → confidence gate → policy → action → audit log.
        - The new audit record is picked up by the WebSocket feed within ~150 ms.
        - Appears in the live dashboard feed identically to a batch transaction.

    Dry-run mode (dry_run=True):
        - Runs classification → confidence gate → policy decision only.
        - Does NOT call any payment/nudge/escalation function.
        - Does NOT write to audit_log.jsonl.
        - Returns the full reasoning with ``simulation_only: True``.
        - The UI displays a SIMULATION badge so it cannot be confused with a live result.
    """
    # Build a synthetic transaction row in the same schema as holdout_set.csv.
    row: dict[str, Any] = {
        "transaction_id": f"manual-{uuid.uuid4().hex[:8]}",
        "amount": body.amount,
        "failure_code": body.failure_code,
        "payment_method": body.payment_method,
        "retry_count_so_far": body.retry_count_so_far,
        "bank_name": body.bank_name,
        # No failure_timestamp — backoff guard will skip the elapsed-hours check.
        "failure_timestamp": None,
        # No ground truth for manual transactions.
        "true_root_cause_category": None,
    }
    try:
        record = process_transaction(row, dry_run=body.dry_run)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Required file missing — did you run train_classifier.py? {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return record


@app.websocket("/ws")
async def audit_ws(websocket: WebSocket) -> None:
    """Stream newly written audit_log.jsonl lines to the browser in real time."""
    await websocket.accept()
    try:
        await follow_audit_log(websocket)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
