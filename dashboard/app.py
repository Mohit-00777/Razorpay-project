"""Live demo dashboard: static SPA + audit WebSocket + on-demand trigger."""

from __future__ import annotations

import subprocess
import sys
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

from orchestrator import process_transaction_id  # noqa: E402
from websocket_feed import follow_audit_log  # noqa: E402

app = FastAPI(
    title="Revenue Recovery Agent",
    version="0.4.0",
    description=(
        "Bounded, explainable, audited payment-failure recovery. "
        "Classify → Policy → Action → Audit."
    ),
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TriggerRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1)


@app.get("/")
async def index() -> FileResponse:
    """Serve the live dashboard SPA."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "audit_log": str(ROOT / "audit" / "audit_log.jsonl")}


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
