"""Alert listing endpoints (TODO Phase 12)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/v1")


@router.get("/alerts")
def list_alerts(request: Request, run_id: str | None = None) -> dict[str, object]:
    records = request.app.state.repository.list_alerts(run_id)
    return {
        "count": len(records),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "run_id": a.run_id,
                "triggered_at_step": a.triggered_at_step,
                "s_n_value": a.s_n_value,
                "root_cause_step": a.root_cause_step,
                "created_at": a.created_at,
            }
            for a in records
        ],
    }
