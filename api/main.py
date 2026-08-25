"""CascadeGuard Core Engine service — FastAPI app (TODO Phase 11).

Wires: trace ingestion -> engine analysis -> persistence -> alert dispatch.
Default composition uses the in-memory repository; set storage.backend=postgres
with CASCADEGUARD_PG_DSN for the real store.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path as _Path
from typing import Any

import yaml
from fastapi import FastAPI

# Allow running as `uvicorn api.main:app` from repo root without install.
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.alerts import AlertDispatcher, build_sink  # noqa: E402
from core.embeddings import load_embedder  # noqa: E402
from core.engine import CascadeGuardEngine  # noqa: E402
from core.graph.pipeline_graph import PipelineGraph  # noqa: E402, F401
from storage.repository import InMemoryTraceRepository  # noqa: E402


def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = path or os.environ.get("CASCADEGUARD_CONFIG", "configs/default.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_engine(config: dict[str, Any]) -> CascadeGuardEngine:
    embedder = load_embedder(config.get("embeddings", {}))
    return CascadeGuardEngine(
        config=config,
        embedder=embedder,
        # generate_fn / entail_fn / counterfactual stack: wire real models here
        # when standing up Track B / SE backends (DECISIONS D-002/D-003).
    )


def create_app(
    config: dict[str, Any] | None = None,
    repository: object | None = None,
    dispatcher: object | None = None,
) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="CascadeGuard Core Engine", version="0.1.0")

    app.state.config = cfg
    app.state.repository = repository or InMemoryTraceRepository()
    sinks = [build_sink(s) for s in cfg.get("alerts", {}).get("sinks", [])]
    app.state.dispatcher = dispatcher or AlertDispatcher(sinks)
    app.state.engine = build_engine(cfg)
    app.state.run_graphs = {}
    app.state.results = {}

    from api.routes.alerts import router as alerts_router
    from api.routes.calibration import router as calibration_router
    from api.routes.ingest import router as ingest_router

    app.include_router(ingest_router)
    app.include_router(alerts_router)
    app.include_router(calibration_router)
    return app


app = create_app()
