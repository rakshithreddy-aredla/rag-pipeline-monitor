"""Phase 9 tests - root-cause localization on multi-hop cascades."""

from core.graph.pipeline_graph import PipelineGraph, Step
from core.localization.root_cause import localize_root_cause


def _chain5():
    """Fig. 1 shape: s1->s2->s3(origin)->s4->s5(alert site)."""
    g = PipelineGraph()
    g.add_step(Step(id="s1", kind="plan", output=""))
    for i in range(2, 6):
        g.add_step(Step(id=f"s{i}", kind="reason", output="", predecessors=[f"s{i-1}"]))
    return g


class TestRootCauseLocalization:
    def test_finds_origin_not_alert_site(self):
        """Corruption born at s3 must outrank its mere carriers s4/s5."""
        g = _chain5()
        alpha = {
            ("s1", "s2"): 0.5,
            ("s2", "s3"): 0.5,
            ("s3", "s4"): 0.9,
            ("s4", "s5"): 0.9,
        }
        hc_s3 = 0.8 + 0.5 * 0.075
        hc_s4 = 0.3 + 0.9 * hc_s3
        hc_s5 = 0.2 + 0.9 * hc_s4
        hc = {"s1": 0.05, "s2": 0.075, "s3": hc_s3, "s4": hc_s4, "s5": hc_s5}

        causes = localize_root_cause(g, alpha, hc, "s5")
        assert causes[0].step_id == "s3"
        # The true origin must outrank the alert site itself (s5 may appear as a
        # candidate with path_length=0, but only below the real origin).
        rank = {c.step_id: i for i, c in enumerate(causes)}
        assert rank.get("s3", 99) < rank.get("s5", 99)

    def test_direct_parent_ranked_when_no_upstream_origin(self):
        """All steps equally clean-ish: nearest high contributor wins."""
        g = _chain5()
        alpha = {("s1", "s2"): 0.3, ("s2", "s3"): 0.3, ("s3", "s4"): 0.3, ("s4", "s5"): 0.3}
        H = {"s1": 0.1, "s2": 0.1, "s3": 0.1, "s4": 0.9, "s5": 0.1}
        hc = {}
        prev = None
        for nid in ["s1", "s2", "s3", "s4", "s5"]:
            hc[nid] = H[nid] + (alpha[(prev, nid)] * hc[prev] if prev else 0.0)
            prev = nid
        causes = localize_root_cause(g, alpha, hc, "s5")
        assert causes[0].step_id == "s4"  # the step that actually injected risk
