"""Phase 6/10/11 integration tests - engine end-to-end, SDK recorder, API."""

from core.embeddings import HashingEmbedder
from core.engine import CascadeGuardEngine
from eval.synthetic import make_paper_fig1_trace


def _engine(config=None):
    # Fake NLI hook: default track is nli_proxy; without any entailment fn wired,
    # the engine correctly REFUSES to analyze (loud dependency error by design).
    return CascadeGuardEngine(
        config=config or {}, embedder=HashingEmbedder(256), entail_fn=lambda a, b: 0.5
    )


class TestEndToEndAnalysis:
    def test_full_pipeline_produces_inspectable_scores_everywhere(self):
        """Phase 6 exit: H / H_cascade / CHAF / CHRS inspectable at every node."""
        result = _engine().analyze_run(make_paper_fig1_trace(corrupted=True))
        for nid in ["s1_plan", "s2_retrieve", "s3_reason", "s4_tool", "s5_synth"]:
            assert nid in result.h_cascade and nid in result.chaf and nid in result.chrs

    def test_theorem1_holds_on_real_engine_output(self):
        from core.scoring.cascade_score import check_theorem_1

        g = make_paper_fig1_trace(corrupted=True)
        result = _engine().analyze_run(g)
        report = check_theorem_1(g, result.chaf)
        assert report.violations == []

    def test_alpha_edges_all_tagged_with_method(self):
        """AGENTS.md 2.3: every alpha carries its estimation track."""
        result = _engine().analyze_run(make_paper_fig1_trace())
        assert len(result.alpha_edges) > 0
        for res in result.alpha_edges.values():
            assert res.method in ("attention", "nli_proxy", "counterfactual")

    def test_semantic_entropy_absent_without_generation_hook(self):
        """No LLM hook wired -> SE never computed anywhere (cheap fidelity only)."""
        result = _engine().analyze_run(make_paper_fig1_trace())
        assert result.semantic_entropy_values == {}
        assert all(f == "cheap" for f in result.monitored_fidelity.values())

    def test_se_runs_only_on_selected_steps(self):
        """Phase 5 exit criterion: SE demonstrably not invoked on unmonitored steps."""
        calls = []

        def gen(prompt):
            calls.append(prompt)
            return "sample answer"

        engine = _engine()
        engine.generate_fn = gen
        engine.entail_fn = lambda a, b: 1.0
        graph = make_paper_fig1_trace()
        # pre-seed placement so exactly ONE step is selected for full fidelity
        engine._placement_cache["paper_fig1"] = {"s3_reason"}
        result = engine.analyze_run(graph)
        assert set(result.semantic_entropy_values.keys()) == {"s3_reason"}
        assert result.monitored_fidelity["s3_reason"] == "full"


class TestSdkRecorder:
    def test_recorder_finalizes_valid_graph(self):
        from sdk.step_recorder import StepRecorder

        rec = StepRecorder("demo-template")
        rec.record_step("a", "plan", output="planning the query about Paris France")
        rec.record_step("b", "retrieve", output="Paris is in France", predecessors=["a"])
        rec.record_step("c", "synthesize", output="Answer: Paris is in France", predecessors=["b"])
        g = rec.finalize()
        assert len(g.nodes) == 3 and g.topo_order()[0] == "a"
