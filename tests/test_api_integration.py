"""API + storage smoke tests (TODO Phase 11/12 v1 scope)."""

import pytest

from api.main import create_app, load_config
from core.alerts import AlertDispatcher, LogSink
from core.engine import CascadeGuardEngine
from eval.synthetic import make_paper_fig1_trace


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    sink = LogSink()
    app = create_app(config=load_config(), dispatcher=AlertDispatcher(sinks=[sink]))
    # Swap in an engine with a fake NLI hook - no model downloads in CI.
    from core.embeddings import HashingEmbedder

    app.state.engine = CascadeGuardEngine(
        config=app.state.config,
        embedder=HashingEmbedder(256),
        entail_fn=lambda a, b: 0.5,
    )
    return TestClient(app), sink


class TestApiEndToEnd:
    def test_run_step_analyze_alert_flow(self, client):
        """Agent step -> API -> engine -> stored scores -> alert reachable."""
        http, sink = client
        created = http.post("/v1/runs", json={"pipeline_template_id": "paper_fig1"}).json()
        run_id = created["run_id"]

        graph = make_paper_fig1_trace(corrupted=True)
        for nid in graph.topo_order():
            step = graph.nodes[nid]
            body = {
                "step_id": step.id,
                "kind": step.kind,
                "output": step.output,
                "retrieved_docs": [
                    {"doc_id": d.doc_id, "content": d.content} for d in step.retrieved_docs
                ],
                "predecessors": step.predecessors,
            }
            r = http.post(f"/v1/runs/{run_id}/steps", json=body)
            assert r.status_code == 200, r.text

        analysis = http.post(f"/v1/runs/{run_id}/analyze").json()
        assert set(analysis["scores"].keys()) == set(graph.nodes.keys())
        assert analysis["scores"]["s3_reason"]["H"] > 0.5  # planted corruption visible

        alerts = http.get("/v1/alerts", params={"run_id": run_id}).json()
        assert alerts["count"] >= 1
        assert alerts["alerts"][0]["root_cause_step"] is not None

        cheap = http.post(f"/v1/cheap_check/{run_id}/s5_synth")
        assert cheap.status_code == 200  # sub-12ms cached lookup contract
        assert "chrs" in cheap.json()
        assert len(sink.sent) == 1  # exactly one alert reached the sink


def test_dispatcher_fires_exactly_once_per_run():
    """TODO Phase 12: no duplicate/spam alerts from repeated CUSUM checks."""
    from uuid import uuid4

    from core.engine import Alert

    sink = LogSink()
    dispatcher = AlertDispatcher(sinks=[sink])
    rid = str(uuid4())
    a1 = Alert(alert_id=str(uuid4()), run_id=rid, triggered_at_step="s5", s_n_value=1.0)
    a2 = Alert(alert_id=str(uuid4()), run_id=rid, triggered_at_step="s5", s_n_value=2.0)
    assert dispatcher.dispatch([a1, a2]) == 1  # second alert on same run suppressed
    assert dispatcher.dispatch([a1]) == 0  # re-dispatch also suppressed
    assert len(sink.sent) == 1
