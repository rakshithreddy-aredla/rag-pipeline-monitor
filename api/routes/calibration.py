"""Calibration + placement refresh endpoints (README §4.7/§4.8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/v1")


class CalibrateIn(BaseModel):
    clean_chrs_values: list[float]


@router.post("/calibration/cusum")
def calibrate_cusum(payload: CalibrateIn, request: Request) -> dict[str, float]:
    engine = request.app.state.engine
    mu0, sigma = engine.calibrate(payload.clean_chrs_values)
    return {"mu0": mu0, "sigma": sigma}


class RefreshIn(BaseModel):
    template_id: str


@router.post("/placement/refresh")
def refresh_placement(payload: RefreshIn, request: Request) -> dict[str, object]:
    graph = request.app.state.run_graphs.get(payload.template_id)
    if graph is None or not graph.nodes:
        raise HTTPException(404, f"unknown or empty template graph {payload.template_id}")
    s_star = request.app.state.engine.refresh_placement(payload.template_id, graph)
    return {"template_id": payload.template_id, "S_star": sorted(s_star)}
