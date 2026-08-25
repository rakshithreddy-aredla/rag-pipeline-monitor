"""Phase 3 tests - propagation Eq. 6 (Track A) and Track B surrogates."""

import numpy as np

from core.scoring.propagation import (
    AlphaResult,
    NliProxyEstimator,
    propagation_coefficient_attention,
)


class TestEq6Attention:
    def test_full_attention_on_span_gives_one(self):
        """Hand-computed: every row puts ALL mass on the span -> ratios [1,1] -> alpha=1."""
        A = np.array([[1.0, 0.0], [1.0, 0.0]])
        res = propagation_coefficient_attention(A, source_token_idx=[0])
        assert res.value == 1.0 and res.method == "attention"

    def test_per_row_ratios_averaged_hand_computed(self):
        """Eq. 6 is per-row ratio then mean: rows give 0.75 and 0.25 -> 0.5."""
        A = np.array([[0.75, 0.25], [0.25, 0.75]])
        res = propagation_coefficient_attention(A, source_token_idx=[0])
        assert abs(res.value - 0.5) < 1e-9

    def test_zero_mass_row_counts_as_zero_ratio(self):
        """Eq. 6 averages over all L_j rows; a dead row's ratio is 0 -> (0.5+0)/2."""
        A = np.array([[0.5, 0.5], [0.0, 0.0]])
        res = propagation_coefficient_attention(A, source_token_idx=[1])
        assert abs(res.value - 0.25) < 1e-9

    def test_no_span_gives_zero(self):
        A = np.ones((2, 2))
        assert propagation_coefficient_attention(A, []).value == 0.0


class TestTrackB:
    def test_nli_proxy_tags_and_clamps(self):
        est = NliProxyEstimator(lambda premise, hyp: 0.9)
        res = est("source claim text", "target reusing the claim")
        assert isinstance(res, AlphaResult)
        assert res.value == 0.9 and res.method == "nli_proxy"

    def test_counterfactual_divergence_hand_computed(self):
        """Orthogonal fake embeddings: divergence 1-cos = 1.0."""
        v1, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        from core.scoring.propagation import CounterfactualEstimator

        stack = CounterfactualEstimator(
            counterfactual_fn=lambda masked_prompt: "completely different output",
            embed_fn=lambda text: (v1 if "original" in text else v2),
            build_masked_prompt=lambda ctx, src: "masked original",
        )
        res = stack("o_i", "context", "original output")
        assert abs(res.raw_value - 1.0) < 1e-9
        assert res.method == "counterfactual"
