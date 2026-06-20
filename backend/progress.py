"""Lightweight in-process progress tracker for long operations.

This is a single-user local demo, so one job runs at a time and a module-level
state is sufficient. Pipeline code calls `set_stage(...)` as it works; the
frontend polls GET /api/progress while a request is in flight.

(For a multi-user deployment this would become per-job state keyed by a job id,
or an SSE/WebSocket stream — noted as a future upgrade.)
"""
from __future__ import annotations

import time

_STATE = {"stage": "idle", "detail": "", "pct": None, "ts": 0.0}


def set_stage(stage: str, detail: str = "", pct: float | None = None) -> None:
    _STATE.update(stage=stage, detail=detail, pct=pct, ts=time.time())


def get_progress() -> dict:
    return dict(_STATE)


def reset() -> None:
    set_stage("idle", "", None)
