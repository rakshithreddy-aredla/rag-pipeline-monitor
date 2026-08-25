"""Semantic Entropy Module — Eq. 9 (Farquhar et al., as cited by the paper).

AGENTS.md §2.4: SE costs K extra LLM calls and runs ONLY on steps selected by
the Monitor Placement Optimizer's S*. The guard below is enforced at the call
site, not just documented (TODO Phase 5).
Clustering is bidirectional NLI entailment — NOT naive embedding k-means — to
stay faithful to the semantic-equivalence-class definition.
"""

from __future__ import annotations

import math
from collections.abc import Callable

EntailmentFn = Callable[[str, str], float]  # (premise, hypothesis) -> P(entailment)


class UnmonitoredStepError(PermissionError):
    """SE was requested for a step outside S* — budget-constrained placement violated."""


def cluster_by_meaning(
    samples: list[str],
    entail_fn: EntailmentFn,
    threshold: float = 0.85,
) -> list[list[int]]:
    """Group sample indices into semantic-equivalence classes.

    a, b cluster together iff entail(a→b) >= threshold AND entail(b→a) >=
    threshold (bidirectional entailment). Greedy single-pass union into the
    first accepting cluster; K is small (≤ ~10) so O(K²) pairwise calls are fine.
    """
    clusters: list[list[int]] = []
    for idx, sample in enumerate(samples):
        placed = False
        for cluster in clusters:
            rep = samples[cluster[0]]
            if entail_fn(rep, sample) >= threshold and entail_fn(sample, rep) >= threshold:
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return clusters


def semantic_entropy(
    step_id: str,
    prompt: str,
    generate_fn: Callable[[str], str],
    entail_fn: EntailmentFn,
    selected_steps: set[str],
    K: int = 10,
    cluster_threshold: float = 0.85,
) -> float:
    """Eq. 9: SE(s_i) = −Σ p(C)·log p(C) over K-sample semantic clusters.

    Raises UnmonitoredStepError unless `step_id` ∈ `selected_steps` (the S* from
    greedy_monitor_placement). This makes it impossible to "accidentally" run
    SE everywhere — that would defeat the paper's §II.H budget design.
    """
    if step_id not in selected_steps:
        raise UnmonitoredStepError(
            f"step {step_id!r} is not in the monitor-placement selection S*; "
            "semantic entropy must not run unconditionally (AGENTS.md §2.4)"
        )
    samples = [generate_fn(prompt) for _ in range(K)]
    clusters = cluster_by_meaning(samples, entail_fn, cluster_threshold)
    probs = [len(c) / K for c in clusters]
    return -sum(p * math.log(p) for p in probs if p > 0)
