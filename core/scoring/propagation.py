"""Propagation Estimator — α_ij, Eq. 6 (Track A) and surrogate tracks (Track B).

AGENTS.md §2.3: every computed α_ij carries its estimation track (`alpha_method`)
— 'attention' | 'nli_proxy' | 'counterfactual'. Never blend or silently substitute.

PAPER-CHECK: the exact normalization of Eq. 6 below follows README §4.3's
transcription (numerator over the attended span, denominator L_j · total mass).
The numerator indexing interprets `context_token_idx` as o_i's token span
inside c_j and `output_token_idx` as the output-token rows doing the attending.
Re-verify against the paper PDF text before the eval phase freezes numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.graph.pipeline_graph import cosine_similarity


class PropagationDependencyError(RuntimeError):
    """Raised when a track needs a model hook that wasn't provided."""


@dataclass(frozen=True)
class AlphaResult:
    value: float  # clamped to [0, clamp_max]
    raw_value: float  # pre-clamp, for detecting estimator bugs (>1 means broken)
    method: str  # 'attention' | 'nli_proxy' | 'counterfactual' — AGENTS.md §2.3


def propagation_coefficient_attention(
    A_j: np.ndarray,
    source_token_idx: list[int],
) -> AlphaResult:
    """Track A — exact Eq. 6 via cross-attention matrix A^(j) of step s_j.

    Eq. 6: α_ij = (1/L_j) · Σ_{ℓ=1..L_j} [ Σ_{k∈idx(o_i)} A^(j)_ℓk ]
                                           / [ Σ_{k'=1..|c_j|} A^(j)_ℓk' ]
    i.e. the per-row attention fraction falling on o_i's token span, averaged
    over ALL L_j output-token rows. `source_token_idx` = idx(o_i): o_i's token
    positions inside c_j (columns).
    """
    if A_j.ndim != 2:
        raise ValueError("attention matrix must be 2-D [L_j, |c_j|]")
    if not source_token_idx:
        return AlphaResult(0.0, 0.0, "attention")
    span_mass = A_j[:, source_token_idx].sum(axis=1)
    row_totals = A_j.sum(axis=1)
    ratios = np.divide(span_mass, row_totals, out=np.zeros_like(span_mass), where=row_totals > 0)
    raw = float(ratios.mean())  # verified verbatim against the paper (Def. 2)
    return AlphaResult(float(np.clip(raw, 0.0, 1.0)), raw, "attention")


# --- Track B -----------------------------------------------------------------

EntailmentFn = Callable[[str, str], float]  # (premise, hypothesis) -> P(entailment)


class NliProxyEstimator:
    """Track B / 'nli_proxy': entailment probability from o_i to o_j as the
    proxy for how strongly s_j's output leans on s_i's claim."""

    def __init__(self, entail_fn: EntailmentFn) -> None:
        self.entail_fn = entail_fn

    def __call__(self, source_output: str, target_output: str) -> AlphaResult:
        raw = float(self.entail_fn(source_output, target_output))
        return AlphaResult(float(np.clip(raw, 0.0, 1.0)), raw, "nli_proxy")


CounterfactualFn = Callable[[str], str]  # masked-context prompt -> counterfactual output


class CounterfactualEstimator:
    """Track B / 'counterfactual': re-run s_j with o_i masked out of c_j;
    embedding divergence between original and counterfactual outputs estimates
    reliance on o_i. Costs one extra LLM call — gated by `expensive_mode`
    (README §4.3); never run unconditionally."""

    def __init__(
        self,
        counterfactual_fn: CounterfactualFn,
        embed_fn: Callable[[str], np.ndarray],
        build_masked_prompt: Callable[[str, str], str],
    ) -> None:
        self.counterfactual_fn = counterfactual_fn
        self.embed_fn = embed_fn
        self.build_masked_prompt = build_masked_prompt

    def __call__(self, source_output: str, target_context: str, target_output: str) -> AlphaResult:
        masked = self.build_masked_prompt(target_context, source_output)
        cf_output = self.counterfactual_fn(masked)
        v_orig = self.embed_fn(target_output)
        v_cf = self.embed_fn(cf_output)
        raw = 1.0 - cosine_similarity(v_orig, v_cf)
        return AlphaResult(float(np.clip(raw, 0.0, 1.0)), raw, "counterfactual")


TRACKS = ("attention", "nli_proxy", "counterfactual")
