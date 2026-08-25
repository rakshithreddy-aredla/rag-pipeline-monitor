"""Phase 5-6 tests - semantic entropy Eq. 9, budget guard, CHRS Eq. 10."""

import math

import pytest

from core.scoring.risk_combiner import chrs
from core.scoring.semantic_entropy import (
    UnmonitoredStepError,
    cluster_by_meaning,
    semantic_entropy,
)


def _never_entails(a, b):
    return 0.0


def _always_entails(a, b):
    return 1.0


class TestEq9SemanticEntropy:
    def test_fully_consistent_samples_give_zero(self):
        samples = ["the tower is in Paris"] * 10
        se = semantic_entropy(
            "s1",
            "prompt",
            lambda p: samples[0],
            _always_entails,
            selected_steps={"s1"},
            K=10,
        )
        assert se == 0.0  # one cluster, p=1 -> -1*log(1)=0

    def test_maximally_diverse_samples_near_log_K(self):
        """K mutually non-entailing answers -> K singleton clusters -> SE=log(K)."""
        K = 8
        counter = {"n": 0}

        def gen(prompt):
            counter["n"] += 1
            return f"distinct answer variant {counter[chr(110)]}"

        se = semantic_entropy(
            "s1",
            "prompt",
            gen,
            _never_entails,
            selected_steps={"s1"},
            K=K,
        )
        assert abs(se - math.log(K)) < 1e-9

    def test_budget_guard_blocks_unmonitored_steps_at_call_site(self):
        """TODO Phase 5 exit: SE must demonstrably NOT run outside S*."""
        with pytest.raises(UnmonitoredStepError):
            semantic_entropy(
                "unmonitored_step",
                "p",
                lambda p: "x",
                _always_entails,
                selected_steps={"some_other_step"},
                K=2,
            )

    def test_bidirectional_entailment_required_for_cluster_join(self):
        """a entails b but not vice versa -> they stay in DIFFERENT clusters."""

        def one_way(a, b):
            return 0.99 if a == "A" else 0.1

        clusters = cluster_by_meaning(["A", "B"], one_way, threshold=0.85)
        assert len(clusters) == 2


class TestEq10Chrs:
    def test_hand_computed_convex_combination(self):
        assert abs(chrs(0.5, 0.25, lam=0.6) - (0.6 * 0.5 + 0.4 * 0.25)) < 1e-12

    def test_lambda_bounds_enforced(self):
        with pytest.raises(ValueError):
            chrs(0.5, 0.5, lam=1.5)
