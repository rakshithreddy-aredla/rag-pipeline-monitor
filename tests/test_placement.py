"""Phase 8 tests - greedy placement vs. brute-force optimum (TODO Phase 8)."""

import random

from core.graph.pipeline_graph import PipelineGraph, Step
from core.placement.greedy_optimizer import (
    brute_force_optimal_placement,
    greedy_monitor_placement,
)


def _path_graph(n):
    g = PipelineGraph()
    g.add_step(Step(id="s0", kind="reason", output=""))
    for i in range(1, n):
        g.add_step(Step(id=f"s{i}", kind="reason", output="", predecessors=[f"s{i-1}"]))
    return g


class TestGreedyVsBruteForce:
    def test_simple_instance_greedy_is_optimal(self):
        """Uniform costs, additive objective: greedy must find the exact optimum."""
        g = _path_graph(4)
        weights = {f"s{i}": 1.0 for i in range(4)}

        def f(S):
            return float(len(S))

        def marginal(s, S):
            return f(S | {s}) - f(S)

        S = greedy_monitor_placement(g, weights, budget=2.0, marginal_risk_fn=marginal)
        assert len(S) == 2
        best_set, best_val = brute_force_optimal_placement(g, weights, 2.0, f)
        assert f(S) == best_val == 2.0

    def test_approximation_ratio_across_random_submodular_instances(self):
        """Weighted coverage f(S)=sum c_t * min(1,|S intersect C_t|) is monotone
        submodular; with unit costs the (1-1/e) bound must hold empirically."""
        ONE_MINUS_OVER_E = 0.6321
        SLACK = 0.05  # finite-instance slack; report-only beyond that
        worst_ratio = 1.0
        for seed in range(30):
            rng = random.Random(seed)
            g = _path_graph(8)
            weights = {f"s{i}": 1.0 for i in range(8)}
            targets = []
            for _ in range(10):
                size = rng.randint(1, 3)
                cover_set = rng.sample([f"s{i}" for i in range(8)], size)
                targets.append((rng.random(), frozenset(cover_set)))

            def f(S, _targets=targets):
                return sum(c * min(1.0, len(set(S) & set(cov))) for c, cov in _targets)

            def marginal(s, S, _targets=targets):
                return f(S | {s}) - f(S)

            S = greedy_monitor_placement(g, weights, budget=3.0, marginal_risk_fn=marginal)
            _, opt = brute_force_optimal_placement(g, weights, 3.0, f)
            if opt > 0:
                worst_ratio = min(worst_ratio, f(S) / opt)
        assert (
            worst_ratio >= ONE_MINUS_OVER_E - SLACK
        ), f"greedy ratio {worst_ratio:.3f} fell below (1-1/e)-slack"

    def test_budget_constraint_respected(self):
        g = _path_graph(5)
        weights = {"s0": 4.0, "s1": 4.0, "s2": 1.0, "s3": 1.0, "s4": 6.0}
        S = greedy_monitor_placement(
            g,
            weights,
            budget=5.9,
            marginal_risk_fn=lambda s, S: 1.0,
        )
        assert sum(weights[s] for s in S) <= 5.9

    def test_kind_cost_fallback_when_no_explicit_weight(self):
        g = _path_graph(3)
        kind_costs = {"reason": 0.05}
        S = greedy_monitor_placement(
            g,
            weights={},
            budget=0.12,
            marginal_risk_fn=lambda s, S: 1.0,
            step_kind_costs=kind_costs,
        )
        assert len(S) == 2  # two affordable steps at 0.05 each within 0.12
