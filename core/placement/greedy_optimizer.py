"""Monitor Placement Optimizer — budget-constrained S* (paper §II.H, Eq. 13).

NP-hard (Theorem 2) → greedy submodular approximation with the (1−1/e) ≈ 63%
guarantee (Nemhauser/Wolsey/Fisher). Runs OFFLINE per pipeline template
(nightly refresh); the live path only looks up the precomputed S*.
"""

from __future__ import annotations

from collections.abc import Callable

from core.graph.pipeline_graph import PipelineGraph


def greedy_monitor_placement(
    graph: PipelineGraph,
    weights: dict[str, float],
    budget: float,
    marginal_risk_fn: Callable[[str, set[str]], float],
    step_kind_costs: dict[str, float] | None = None,
) -> set[str]:
    """Greedy Δrisk(s)/w(s) selection under budget B — README §4.8 pseudocode.

    `weights[s]` is the empirical monitoring cost of step s; when a step has no
    explicit weight, fall back to its kind's calibrated cost (`step_kind_costs`)
    rather than assuming uniform cost.
    """

    def cost_of(s: str) -> float:
        if s in weights:
            return weights[s]
        if step_kind_costs and graph.nodes[s].kind in step_kind_costs:
            return step_kind_costs[graph.nodes[s].kind]
        raise KeyError(f"no monitoring cost for step {s!r}")

    selected: set[str] = set()
    remaining = budget
    candidates = set(graph.nodes)
    while candidates:
        best: str | None = None
        best_ratio = -1.0
        for s in sorted(candidates):  # sorted for determinism across runs
            w = cost_of(s)
            if w > remaining or w <= 0:
                continue
            ratio = marginal_risk_fn(s, selected) / w
            if ratio > best_ratio:
                best, best_ratio = s, ratio
        if best is None:
            break  # nothing affordable remains
        selected.add(best)
        remaining -= cost_of(best)
        candidates.remove(best)
    return selected


def brute_force_optimal_placement(
    graph: PipelineGraph,
    weights: dict[str, float],
    budget: float,
    objective_fn: Callable[[set[str]], float],
    max_nodes: int = 12,
) -> tuple[set[str], float]:
    """Exhaustive optimum over all budget-feasible subsets — validation oracle.

    Only usable on small DAGs (≤ max_nodes). Returns (S*, best objective value).
    Used by tests to confirm greedy stays within the theoretical bound.
    """
    if len(graph.nodes) > max_nodes:
        raise ValueError(f"brute force limited to {max_nodes} nodes")
    node_ids = list(graph.nodes)
    best_set: set[str] = set()
    best_value = 0.0
    for mask in range(1 << len(node_ids)):
        subset = {node_ids[i] for i in range(len(node_ids)) if mask >> i & 1}
        if sum(weights.get(s, 0.0) for s in subset) > budget:
            continue
        value = objective_fn(subset)
        if value > best_value:
            best_value, best_set = value, subset
    return best_set, best_value
