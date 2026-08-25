"""Phase 2 tests - faithfulness Eq. 3-4 with hand-checkable expectations."""

import numpy as np

from core.embeddings import HashingEmbedder
from core.graph.pipeline_graph import Document, cosine_similarity
from core.scoring.faithfulness import (
    chunk_average_embedding,
    faithfulness,
    hallucination_score,
)

emb = HashingEmbedder(256)
NL = chr(10)


def test_identical_output_and_context_gives_H_near_zero():
    text = "The Eiffel Tower is located in Paris, France."
    h = hallucination_score(text, [Document("d1", text)], emb)
    assert h is not None and h < 0.05  # near-identical -> H near 0


def test_unrelated_output_and_context_gives_high_H():
    out = (
        "zebra quantum xkcd vulture plinth marmalade trombone cyclone wax "
        "fjord labyrinth obsidian marigold tumbleweed gargoyle zephyr quiver"
    )
    ctx = (
        "quarterly fiscal revenue projections spreadsheet depreciation ledger "
        "audit compliance invoice payroll expenditure forecast balance treasury"
    )
    h = hallucination_score(out, [Document("d1", ctx)], emb)
    assert h > 0.6


def test_no_retrieval_context_returns_none_not_zero():
    assert faithfulness("anything", [], emb) is None


def test_multi_chunk_equals_mean_of_chunk_embeddings():
    """Hand-computed: docs forced into 2 chunks must equal mean-of-chunks."""
    d1 = Document("d1", "first document about towers in Paris France")
    d2 = Document("d2", "second document entirely separate words about Berlin")
    out = "summary mentioning Paris"
    got = chunk_average_embedding([d1, d2], emb, max_concat_chars=20)
    manual = np.mean([emb(d1.content), emb(d2.content)], axis=0)
    assert np.allclose(got, manual)
    expected = cosine_similarity(emb(out), manual)
    assert abs(faithfulness(out, [d1, d2], emb, max_concat_chars=20) - expected) < 1e-9


def test_short_set_is_single_chunk_of_joined_docs():
    docs = [Document("d1", "alpha"), Document("d2", "beta")]
    got = chunk_average_embedding(docs, emb, max_concat_chars=6000)
    assert np.allclose(got, emb("alpha" + NL + "beta"))
