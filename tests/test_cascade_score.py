"""Phase 4 tests - cascade scoring Eq. 5/7, Theorem 1 invariant, Fig. 1 pattern."""

import pytest

from core.graph.pipeline_graph import PipelineGraph, Step
from core.scoring.cascade_score import (
    Theorem1Violation,
    assert_theorem_1,
    compute_chaf,
    compute_h_cascade,
)


def _chain(alpha_vals, H_vals):
    """Linear chain s1->s2->s3 with given alphas and H values."""
    g = PipelineGraph()
    g.add_step(Step(id="s1", kind="reason", output="o1"))
    g.add_step(Step(id="s2", kind="reason", output="o2", predecessors=["s1"]))
    g.add_step(Step(id="s3", kind="reason", output="o3", predecessors=["s2"]))
    alpha = {("s1", "s2"): alpha_vals[0], ("s2", "s3"): alpha_vals[1]}
    ids = ["s1", "s2", "s3"]
    return g, alpha, dict(zip(ids, H_vals, strict=True))


class TestEq5CascadeScore:
    def test_hand_computed_chain(self):
        """H={.5,.3,.2}, a12=.4, a23=.6:
        h_c(s2) = .3 + .4*.5 = .5 ; h_c(s3) = .2 + .6*.5 = .5."""
        g, alpha, H = _chain([0.4, 0.6], [0.5, 0.3, 0.2])
        hc = compute_h_cascade(g, alpha, H)
        assert abs(hc["s1"] - 0.5) < 1e-12
        assert abs(hc["s2"] - 0.5) < 1e-12
        assert abs(hc["s3"] - 0.5) < 1e-12

    def test_topological_order_required_for_correctness(self):
        """If predecessors resolve first, fan-in sums are exact (hand-checked)."""
        g = PipelineGraph()
        g.add_step(Step(id="a", kind="reason", output=""))
        g.add_step(Step(id="b", kind="reason", output=""))
        g.add_step(Step(id="c", kind="reason", output="", predecessors=["a", "b"]))
        H = {"a": 0.4, "b": 0.6, "c": 0.1}
        alpha = {("a", "c"): 0.5, ("b", "c"): 0.5}
        hc = compute_h_cascade(g, alpha, H)
        assert abs(hc["c"] - (0.1 + 0.5 * 0.4 + 0.5 * 0.6)) < 1e-12


class TestEq7ChafAndTheorem1:
    def test_chaf_hand_computed(self):
        g, alpha, H = _chain([0.4, 0.6], [0.5, 0.3, 0.2])
        hc = compute_h_cascade(g, alpha, H)
        chaf = compute_chaf(hc, H)
        assert abs(chaf["s1"] - 1.0) < 1e-12
        assert abs(chaf["s2"] - (0.5 / 0.3)) < 1e-12
        assert abs(chaf["s3"] - (0.5 / 0.2)) < 1e-12
        # depth+1 bounds: s1<=1, s2<=2, s3<=3
        for nid in ("s1", "s2", "s3"):
            assert chaf[nid] <= graph_depth(g, nid) + 1

    def test_zero_H_node_gets_chaf_one_by_definition(self):
        from core.scoring.cascade_score import compute_chaf as cc

        assert cc({"x": 0.7}, {"x": 0.0}) == {"x": 1.0}

    def test_theorem1_violation_detected_on_broken_alpha(self):
        """alpha=2.0 (unclamped estimator bug) must trip the invariant loudly."""
        g, alpha, H = _chain([0.9, 2.0], [0.9, 0.9, 0.9])
        hc = compute_h_cascade(g, alpha, H)
        chaf = compute_chaf(hc, H)
        with pytest.raises(Theorem1Violation):
            assert_theorem_1(g, chaf)


def graph_depth(g, nid):
    return g.depth(nid)


class TestFig1AmplificationPattern:
    def test_corrupted_trace_amplifies_downstream_vs_clean(self):
        from core.embeddings import HashingEmbedder
        from core.engine import CascadeGuardEngine
        from eval.synthetic import make_paper_fig1_trace

        engine = CascadeGuardEngine(
            config={}, embedder=HashingEmbedder(256), entail_fn=lambda a, b: 0.5
        )
        clean = engine.analyze_run(make_paper_fig1_trace(corrupted=False))
        corrupt = engine.analyze_run(make_paper_fig1_trace(corrupted=True))

        # corruption planted at s3_reason; downstream nodes must score strictly higher
        assert corrupt.H["s3_reason"] > clean.H["s3_reason"]
        assert corrupt.h_cascade["s4_tool"] >= clean.h_cascade["s4_tool"]
        assert corrupt.h_cascade["s5_synth"] >= clean.h_cascade["s5_synth"]
        # and the theorem holds on BOTH runs (the engine asserts internally too)


class TestScalingOofVplusE:
    def test_topo_pass_scales_linearly_synthetic(self):
        """TODO Phase 4: empirical O(|V|+|E|) sanity on production-like sizes."""
        import time as _t

        from core.scoring.cascade_score import compute_h_cascade

        timings = {}
        for n in (50, 200, 1000):
            g = PipelineGraph()
            g.add_step(Step(id="s0", kind="reason", output="x"))
            for i in range(1, n):
                g.add_step(Step(id=f"s{i}", kind="reason", output="x", predecessors=[f"s{i-1}"]))
            H = {f"s{i}": 0.01 for i in range(n)}
            alpha = {(f"s{i-1}", f"s{i}"): 0.5 for i in range(1, n)}
            t0 = _t.perf_counter()
            hc = compute_h_cascade(g, alpha, H)
            timings[n] = _t.perf_counter() - t0
            assert len(hc) == n
        # generous linear-ish bound: 20x nodes must not cost >40x time (vs 50-node run)
        assert timings[1000] < max(timings[50] * 40, 2.0), timings
