"""Risk Combiner — Eq. 10: CHRS(s_i) = λ·H_cascade(s_i) + (1−λ)·SE(s_i).

λ comes from configs/default.yaml (`risk_combiner.lambda`) and is a PLACEHOLDER
until the README §6 cross-validation sweep runs (DECISIONS.md D-001) — never
hardcode it inline.
"""

from __future__ import annotations


def chrs(h_cascade_i: float, se_i: float, lam: float) -> float:
    """Eq. 10. `lam` ∈ [0,1] is λ from config.

    For steps monitored at 'cheap' fidelity (not in S*), pass se_i=0.0 — their
    recorded CHRS then reflects structural risk only, which is why every stored
    row also carries `monitored_fidelity` so eval can separate the populations.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0,1], got {lam}")
    return float(lam * h_cascade_i + (1 - lam) * se_i)
