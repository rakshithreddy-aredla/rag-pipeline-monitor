"""Root-cause localization - README section 5's backward walk.

When an alert fires at s_j, expand Eq. 5 recursively: every node k upstream
receives credit for its OWN risk H(k) propagated along alpha-weighted paths
into the alerted node. Ranking by that attributed mass surfaces the ORIGIN of
a cascade, not merely the carrier steps closest to the alert.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.graph.pipeline_graph import PipelineGraph


@dataclass(frozen=True)
class RootCause:
    step_id: str
    share: float  # fraction of the alerted node's H_cascade attributable to this step's own H
    path_length: int  # hops between root cause and alerted node


def localize_root_cause(
    graph: PipelineGraph,
    alpha: dict[tuple[str, str], float],
    h_cascade: dict[str, float],
    alerted_node: str,
    top_n: int = 3,
    min_share: float = 0.05,
) -> list[RootCause]:
    """Attribute H_cascade(alerted) back to origins via dynamic programming.

    M(j)[k] = alpha-mass of node k's own-H present inside H_cascade(j):
        M(j) = {j: H(j)} + for each pred i: M(j)[k] += alpha_ij * M(i)[k]
    share(k) = M(alerted)[k] / H_cascade(alerted). Computed in reverse
    topological order so each predecessor's attribution map is complete.
    """
    if alerted_node not in graph.nodes:
        raise KeyError(alerted_node)
    if h_cascade.get(alerted_node, 0.0) <= 0:
        return [RootCause(alerted_node, 1.0, 0)]

    attribution: dict[str, dict[str, float]] = {}
    hops_from_alert: dict[str, int] = {alerted_node: 0}

    # Forward topo order: each node's map needs its predecessors' maps,
    # which are always earlier in topological order.
    for node_id in graph.topo_order():
        # Own-H via Eq. 5 inversion: H(j) = H_cascade(j) - sum(alpha_ij * H_cascade(i)).
        # Exact, not an approximation - the propagated summands are already final.
        pred_mass = sum(
            alpha.get((pred, node_id), 0.0) * h_cascade.get(pred, 0.0)
            for pred in graph.edges_into(node_id)
        )
        own = max(h_cascade.get(node_id, 0.0) - pred_mass, 0.0)
        m: dict[str, float] = {node_id: own}
        for pred in graph.edges_into(node_id):
            a = alpha.get((pred, node_id), 0.0)
            if a <= 0 or pred not in attribution:
                continue
            for origin, mass in attribution[pred].items():
                m[origin] = m.get(origin, 0.0) + a * mass
        attribution[node_id] = m

    total = h_cascade[alerted_node]
    candidates: list[RootCause] = []
    for origin, mass in attribution[alerted_node].items():
        if mass <= 0:
            continue
        share = mass / total
        # path_length 0 means the alert site IS the origin - report that
        # honestly rather than pinning the cause on weak upstream noise.
        hops_from_alert.setdefault(origin, _shortest_hops(graph, origin, alerted_node))
        candidates.append(RootCause(origin, share, hops_from_alert[origin]))
    candidates.sort(key=lambda c: (-c.share, c.path_length))
    kept = [c for c in candidates if c.share >= min_share][:top_n]
    # An alert with NO attributable origin is not actionable - always surface
    # at least the single best contributor, even a weak one.
    if not kept and candidates:
        kept = [candidates[0]]
    return kept


def _shortest_hops(graph: PipelineGraph, src: str, dst: str) -> int:
    from collections import deque

    dist = {src: 0}
    q: deque[str] = deque([src])
    while q:
        node = q.popleft()
        if node == dst:
            return dist[node]
        for succ in graph.edges_out_of(node):
            if succ not in dist:
                dist[succ] = dist[node] + 1
                q.append(succ)
    return 10**9
