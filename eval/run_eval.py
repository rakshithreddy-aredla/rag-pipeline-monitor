"""Evaluation harness entry point (README section 6, v1 scope).

Runs the checks needing NO external dataset or model:
  1. Theorem 1 sweep across random DAGs (zero violations allowed)
  2. CUSUM ARL0 simulation vs. the Eq. 12 target (Proposition 1)
  3. Fig. 1 amplification demonstration
External-dataset evaluation (HaluEval/RAGTruth/AgentHallu + baselines + lambda
tuning) stays pending until datasets are fetched; loaders raise with instructions.
"""

from __future__ import annotations

import argparse

import numpy as np

from core.detection.cusum import CusumDetector, CusumParams
from core.embeddings import HashingEmbedder
from core.engine import CascadeGuardEngine
from eval.synthetic import make_paper_fig1_trace, make_random_dag


def _engine() -> CascadeGuardEngine:
    # Placeholder entail_fn until the real NLI model is wired (DECISIONS D-002).
    return CascadeGuardEngine(config={}, embedder=HashingEmbedder(64), entail_fn=lambda a, b: 0.5)


def theorem1_sweep(n_graphs: int = 50) -> dict:
    violations = 0
    max_ratio = 0.0
    for seed in range(n_graphs):
        graph, _ = make_random_dag(seed=seed, n_nodes=8)
        result = _engine().analyze_run(graph)
        for nid, chaf_value in result.chaf.items():
            bound = graph.depth(nid) + 1
            max_ratio = max(max_ratio, chaf_value / bound)
            if chaf_value > bound + 1e-9:
                violations += 1
    return {
        "graphs": n_graphs,
        "violations": violations,
        "max_chaf_over_bound": round(max_ratio, 4),
    }


def cusum_arl0_simulation(n_runs: int = 300, cap: int = 5000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    mu0, sigma, delta = 0.3, 0.4, 0.2  # |Delta|=0.25: inside Eq.12 validity regime
    params = CusumParams(mu0=mu0, sigma=sigma, delta=delta, arl0_target=500)
    lengths = []
    for _ in range(n_runs):
        det = CusumDetector(params)
        fired_at = None
        for step_idx in range(cap):
            x = float(rng.normal(mu0, sigma))
            if det.update(x):
                fired_at = step_idx
                break
        lengths.append(fired_at if fired_at is not None else cap)
    mean_arl = float(np.mean(lengths))
    return {
        "runs": n_runs,
        "target": params.arl0_target,
        "empirical_mean_ARL0": round(mean_arl, 1),
        "median_steps_to_fire": float(np.median(lengths)),
        "censored_at_cap": sum(1 for L in lengths if L == cap),
    }


def fig1_amplification() -> dict:
    engine = CascadeGuardEngine(
        config={}, embedder=HashingEmbedder(256), entail_fn=lambda a, b: 0.5
    )
    clean = engine.analyze_run(make_paper_fig1_trace(corrupted=False))
    corrupt = engine.analyze_run(make_paper_fig1_trace(corrupted=True))
    sink_clean = clean.h_cascade["s5_synth"]
    sink_corrupt = corrupt.h_cascade["s5_synth"]
    causes = []
    if corrupt.alerts:
        causes = [rc.step_id for rc in corrupt.alerts[0].root_causes]
    return {
        "clean_sink_h_cascade": round(sink_clean, 4),
        "corrupt_sink_h_cascade": round(sink_corrupt, 4),
        "amplification_ratio": round(sink_corrupt / max(sink_clean, 1e-9), 2),
        "alert_fired_on_corrupt": corrupt.alerted,
        "root_causes_of_alert": causes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CascadeGuard v1 validation suite")
    parser.add_argument("--graphs", type=int, default=50)
    parser.add_argument("--cusum-runs", type=int, default=300)
    args = parser.parse_args()

    print("== CascadeGuard v1 validation ==")
    t1 = theorem1_sweep(args.graphs)
    print(f"Theorem 1 sweep: {t1}")
    if t1["violations"] > 0:
        print("FAIL: CHAF depth-bound violated - alpha estimation bug")
        raise SystemExit(1)

    arl = cusum_arl0_simulation(n_runs=args.cusum_runs)
    print(f"CUSUM ARL0 simulation: {arl}")

    fig = fig1_amplification()
    print(f"Fig.1 amplification demo: {fig}")


if __name__ == "__main__":
    main()
