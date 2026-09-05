"""Tail audit/audit_log.jsonl and push complete JSON objects to WebSocket clients."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

import sys

ROOT = Path(__file__).resolve().parents[1]
_audit_dir = str(ROOT / "audit")
if _audit_dir not in sys.path:
    sys.path.insert(0, _audit_dir)

from logger import AUDIT_PATH  # noqa: E402

# How often to check the file for new bytes (seconds)
POLL_SECONDS = 0.15


def parse_jsonl_chunk(buffer: str, chunk: str) -> tuple[list[dict[str, Any]], str]:
    """Append *chunk* to *buffer*, parse complete JSONL lines, return leftovers."""
    buffer += chunk.replace("\r\n", "\n")
    records: list[dict[str, Any]] = []
    while True:
        newline = buffer.find("\n")
        if newline < 0:
            break
        line = buffer[:newline].strip()
        buffer = buffer[newline + 1 :]
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, buffer


def envelope(record: dict[str, Any], *, replay: bool) -> dict[str, Any]:
    """Wrap an audit record in a typed message envelope for the browser."""
    return {"type": "audit_record", "replay": replay, "record": record}


async def send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, default=str))


async def follow_audit_log(
    websocket: WebSocket,
    path: Path = AUDIT_PATH,
    poll_seconds: float = POLL_SECONDS,
) -> None:
    """Send existing lines as replay messages, then stream newly appended rows.

    Protocol (browser receives):
        {"type": "audit_record", "replay": true,  "record": {...}}   – existing history
        {"type": "ready",        "replay": true,  "path":  "..."}    – end of replay sentinel
        {"type": "audit_record", "replay": false, "record": {...}}   – live new entries
    """
    offset = 0
    buffer = ""
    replay_done = False
    inode: int | None = None

    try:
        while True:
            if not path.exists():
                # File not yet created — send the ready sentinel once then idle
                if not replay_done:
                    await send_json(
                        websocket,
                        {"type": "ready", "replay": True, "path": str(path)},
                    )
                    replay_done = True
                await asyncio.sleep(poll_seconds)
                continue

            stat = path.stat()
            current_inode = getattr(stat, "st_ino", None)

            # Detect log rotation / truncation
            if stat.st_size < offset or (
                inode is not None and current_inode != inode
            ):
                offset = 0
                buffer = ""

            inode = current_inode

            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()

            records, buffer = parse_jsonl_chunk(buffer, chunk)
            for record in records:
                await send_json(
                    websocket, envelope(record, replay=not replay_done)
                )

            if not replay_done:
                await send_json(
                    websocket,
                    {"type": "ready", "replay": True, "path": str(path)},
                )
                replay_done = True

            await asyncio.sleep(poll_seconds)

    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001
        # Swallow unexpected errors so the server stays up; the browser will reconnect.
        return
