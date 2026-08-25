"""Cascade Score Engine — Eq. 5 (H_cascade) and Eq. 7 (CHAF).

AGENTS.md §2.1/§2.2 invariants live here:
- H_cascade is computed via ONE topological traversal, never per-node recursion.
- CHAF(s_j) <= depth(s_j)+1 must hold for every node (Theorem 1); a violation
  means α_ij estimation is broken — never loosen the assertion to go green.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.graph.pipeline_graph import PipelineGraph


class Theorem1Violation(AssertionError):
    """CHAF exceeded depth+1 on some node — α_ij estimator bug, not a bad theorem."""


def compute_h_cascade(
    graph: PipelineGraph,
    alpha: dict[tuple[str, str], float],
    H: dict[str, float],
) -> dict[str, float]:
    """Eq. 5 in a single topological pass — O(|V|+|E|) (paper §II.C).

    H_cascade(s_j) = H(s_j) + Σ_{(s_i,s_j)∈E} α_ij · H_cascade(s_i)
    Topological order guarantees every predecessor's cascade score is final
    before it is consumed.
    """
    h_cascade: dict[str, float] = {}
    for node_id in graph.topo_order():
        total = H[node_id]
        for pred_id in graph.edges_into(node_id):
            total += alpha[(pred_id, node_id)] * h_cascade[pred_id]
        h_cascade[node_id] = total
    return h_cascade


def compute_chaf(
    h_cascade: dict[str, float],
    H: dict[str, float],
) -> dict[str, float]:
    """Eq. 7: CHAF(s_j) = H_cascade(s_j)/H(s_j); defined as 1.0 when H(s_j)=0."""
    return {n: (h_cascade[n] / H[n]) if H[n] > 0 else 1.0 for n in h_cascade}


@dataclass(frozen=True)
class Theorem1Report:
    violations: list[str]  # node ids where CHAF > depth+1
    max_ratio_observed: float
    checked: int


def check_theorem_1(
    graph: PipelineGraph,
    chaf: dict[str, float],
    tolerance: float = 1e-9,
) -> Theorem1Report:
    """Runtime-testable Theorem 1 invariant (README §4.4).

    Use in tests/staging. A violation indicates an unclamped or mislabeled
    α_ij — fix the estimator, do NOT weaken this check (AGENTS.md §2.2).
    """
    violations: list[str] = []
    max_ratio = 0.0
    for node_id, chaf_value in chaf.items():
        bound = graph.depth(node_id) + 1
        max_ratio = max(max_ratio, chaf_value / bound if bound > 0 else chaf_value)
        if chaf_value > bound + tolerance:
            violations.append(node_id)
    return Theorem1Report(violations=violations, max_ratio_observed=max_ratio, checked=len(chaf))


def assert_theorem_1(graph: PipelineGraph, chaf: dict[str, float]) -> Theorem1Report:
    report = check_theorem_1(graph, chaf)
    if report.violations:
        offenders = ", ".join(
            f"{n}: CHAF={chaf[n]:.4f} > depth({n})+1={graph.depth(n) + 1}"
            for n in report.violations[:5]
        )
        raise Theorem1Violation(
            f"Theorem 1 violated on {len(report.violations)} node(s): {offenders}. "
            "α_ij estimation is broken (likely unclamped >1) — fix the estimator."
        )
    return report
