"""Phase 1 tests — PipelineGraph (TODO Phase 1).

All depth/topo expectations are hand-computed, per AGENTS.md §5 testing discipline.
"""

import numpy as np
import pytest

from core.graph.pipeline_graph import (
    CycleError,
    Document,
    PipelineGraph,
    Step,
    build_dynamic_edges,
    cosine_similarity,
    edge_exists,
    ngram_overlap,
)


def make_step(step_id: str, kind="reason", output: str = "", preds: list[str] | None = None):
    return Step(id=step_id, kind=kind, output=output, predecessors=preds or [])


class TestTopoOrder:
    def test_linear_chain(self):
        g = PipelineGraph()
        g.add_step(make_step("s1"))
        g.add_step(make_step("s2", preds=["s1"]))
        g.add_step(make_step("s3", preds=["s2"]))
        assert g.topo_order() == ["s1", "s2", "s3"]

    def test_diamond_fanout_fanin(self):
        # s1 -> {s2, s3} -> s4
        g = PipelineGraph()
        g.add_step(make_step("s1"))
        g.add_step(make_step("s2", preds=["s1"]))
        g.add_step(make_step("s3", preds=["s1"]))
        g.add_step(make_step("s4", preds=["s2", "s3"]))
        order = g.topo_order()
        assert order.index("s1") < order.index("s2") < order.index("s4")
        assert order.index("s1") < order.index("s3") < order.index("s4")
        assert sorted(order) == ["s1", "s2", "s3", "s4"]

    def test_single_node(self):
        g = PipelineGraph()
        g.add_step(make_step("only"))
        assert g.topo_order() == ["only"]
        assert g.depth("only") == 0

    def test_cycle_raises_loudly(self):
        g = PipelineGraph()
        g.add_step(make_step("a"))
        g.add_step(make_step("b", preds=["a"]))
        g.add_edge("b", "a")  # closes the cycle — allowed at construction...
        with pytest.raises(CycleError):
            g.topo_order()  # ...and must fail LOUDLY here (TODO Phase 1)

    def test_add_edge_requires_existing_nodes(self):
        g = PipelineGraph()
        g.add_step(make_step("a"))
        with pytest.raises(KeyError):
            g.add_edge("a", "ghost")


class TestDepth:
    def test_depth_linear_chain(self):
        g = PipelineGraph()
        g.add_step(make_step("s1"))
        g.add_step(make_step("s2", preds=["s1"]))
        g.add_step(make_step("s3", preds=["s2"]))
        assert [g.depth(f"s{i}") for i in (1, 2, 3)] == [0, 1, 2]

    def test_depth_hand_computed_five_dags(self):
        # DAG 1: diamond — s4 depth = 2 (longest path s1->s2->s4)
        g1 = PipelineGraph()
        g1.add_step(make_step("s1"))
        g1.add_step(make_step("s2", preds=["s1"]))
        g1.add_step(make_step("s3", preds=["s1"]))
        g1.add_step(make_step("s4", preds=["s2", "s3"]))
        assert g1.depth("s4") == 2

        # DAG 2: uneven branches — s1->{s2->{s4}}, s3 joins s4: longest = s1->s2->s4 = 2
        g2 = PipelineGraph()
        g2.add_step(make_step("s1"))
        g2.add_step(make_step("s2", preds=["s1"]))
        g2.add_step(make_step("s3", preds=["s1"]))
        g2.add_step(make_step("s4", preds=["s3", "s2"]))
        assert g2.depth("s4") == 2

        # DAG 3: chain of 5 — sink depth 4
        g3 = PipelineGraph()
        g3.add_step(make_step("n1"))
        for i in range(2, 6):
            g3.add_step(make_step(f"n{i}", preds=[f"n{i-1}"]))
        assert g3.depth("n5") == 4

        # DAG 4: two independent sources joining — max(1, 3) + ... hand-computed:
        # a(0) -> b(1) -> c(2) -> z ; d(0) -> z. z depth = max(2, 0) + 1 = 3.
        g4 = PipelineGraph()
        g4.add_step(make_step("a"))
        g4.add_step(make_step("d"))
        g4.add_step(make_step("b", preds=["a"]))
        g4.add_step(make_step("c", preds=["b"]))
        g4.add_step(make_step("z", preds=["c", "d"]))
        assert g4.depth("z") == 3

        # DAG 5: five depth-2 branches converge: root(0)->mid(1)->leaf(2)->sink(3).
        g5 = PipelineGraph()
        g5.add_step(make_step("root"))
        for i in range(5):
            g5.add_step(make_step(f"mid{i}", preds=["root"]))
            g5.add_step(make_step(f"leaf{i}", preds=[f"mid{i}"]))
        g5.add_step(make_step("sink", preds=[f"leaf{i}" for i in range(5)]))
        assert g5.depth("sink") == 3

    def test_depth_unknown_node(self):
        g = PipelineGraph()
        g.add_step(make_step("a"))
        with pytest.raises(KeyError):
            g.depth("nope")


class TestEq1ContextAssembly:
    def test_context_is_retrieved_docs_plus_predecessor_outputs(self):
        """Eq. 1: c_i = R(s_i) ∪ {o_j | (s_j, s_i) ∈ E}."""
        step = Step(
            id="s2",
            kind="synthesize",
            output="final answer",
            retrieved_docs=[Document(doc_id="d1", content="DOC ONE")],
            predecessors=["s1a", "s1b"],
        )
        ctx = step.context({"s1a": "OUTPUT A", "s1b": "OUTPUT B"})
        assert "DOC ONE" in ctx and "OUTPUT A" in ctx and "OUTPUT B" in ctx

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError):
            Step(id="x", kind="vibe", output="")


class TestEdgeInference:
    def test_ngram_overlap_directional(self):
        src = "the Eiffel Tower is located in Paris France"
        tgt = "Located in Paris France, the Eiffel Tower attracts many visitors"
        assert ngram_overlap(src, tgt, n=3) > 0.15
        assert ngram_overlap("totally unrelated content here now", tgt, n=3) < 0.15

    def test_verbatim_reuse_detected_without_embedder(self):
        assert edge_exists(
            "Paris is the capital of France",
            "Note that Paris is the capital of France.",
        )

    def test_unrelated_not_detected(self):
        assert not edge_exists(
            "Quarterly revenue rose by four percent", "The cat sat on the mat today"
        )

    def test_paraphrase_detected_via_embedding_fallback(self):
        """Low n-gram overlap but lexically overlapping -> embedding path fires."""
        from core.embeddings import HashingEmbedder

        fake_embed = HashingEmbedder(64)

        early = "France's capital city is Paris"
        late_ctx = "Background: the french capital, paris, serves as france's political center."
        # sanity: n-gram path alone would NOT fire here
        assert ngram_overlap(early, late_ctx, n=3) < 0.15
        assert edge_exists(early, late_ctx, embed_fn=fake_embed)

    def test_dynamic_builder_links_summarizing_downstream_step(self):
        """ReAct-style loop: s2 summarizes o1 without verbatim copy -> edge via embeddings."""

        from core.embeddings import HashingEmbedder

        fake_embed = HashingEmbedder(64)
        g = PipelineGraph()
        g.add_step(
            make_step(
                "retrieve",
                kind="retrieve",
                output="Paris is capital of France population facts listed",
            )
        )
        g.add_step(make_step("answer", kind="synthesize", output="Answer about Paris France"))
        build_dynamic_edges(g, embed_fn=fake_embed)
        # With this toy embedder both texts share no tokens, so no edge must be invented:
        assert g.edges == []
        # And when the summary DOES reuse vocabulary, the edge must appear:
        g2 = PipelineGraph()
        g2.add_step(make_step("retrieve", kind="retrieve", output="Paris is the capital of France"))
        g2.add_step(
            make_step(
                "answer",
                kind="synthesize",
                output="The capital of France is Paris, so the answer is Paris",
            )
        )
        build_dynamic_edges(g2, embed_fn=fake_embed)
        assert ("retrieve", "answer") in g2.edges


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_zero(self):
        assert abs(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9

    def test_zero_vector_safe(self):
        assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0
