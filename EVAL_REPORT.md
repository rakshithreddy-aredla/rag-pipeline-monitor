# EVAL_REPORT.md

Results of running CascadeGuard's evaluation plan (README §6, TODO Phase 14) against the paper's claims. This file starts as a template — fill it in as Phase 14 actually runs, with real numbers. Don't fill in numbers you haven't actually measured; leave sections marked `PENDING` until they're real. A half-honest eval report is worse than an incomplete one, especially since this is the document you'll most likely end up walking someone through in an interview.

---

## What the paper claims (baseline to reproduce)

| Claim | Value |
|---|---|
| Reduction in undetected cascading failures vs. best single-step baseline | 34.7% |
| Latency overhead | 12 ms per pipeline step |
| Datasets | HaluEval-RAG, RAGTruth, AgentHallu |
| Target false-alarm rate (from ARL0=500, Eq. 12) | < 0.2% per pipeline execution |
| Monitor placement approximation guarantee | (1 − 1/e) ≈ 63% of optimal |

---

## 1. Dataset preparation status

| Dataset | Loader implemented | # traces converted | Ground-truth labels available |
|---|---|---|---|
| HaluEval-RAG | PENDING | — | — |
| RAGTruth | PENDING | — | — |
| AgentHallu | PENDING | — | — |

Notes on conversion difficulties, label quality issues, or dataset-specific quirks go here.

---

## 2. Baselines implemented

| Baseline | Status | Source |
|---|---|---|
| RAGAS | PENDING | (OSS wrap / reimplementation — note which) |
| SelfCheckGPT | PENDING | (OSS wrap / reimplementation — note which) |
| RAGTruth / FActScore / ARES | Not reimplemented — cited from published numbers | paper references [4],[6],[7] |

---

## 3. λ tuning (TODO Phase 14, decision → DECISIONS.md D-001)

- **Search method:** PENDING (grid / Bayesian)
- **Search range:** PENDING
- **Validation split:** PENDING
- **Optimization target:** F1 on detection (per README §6 step 4)
- **Winning λ:** PENDING
- **Full sweep results:** (table or link to raw results file)

---

## 4. Headline results — CascadeGuard vs. baselines

| Metric | RAGAS | SelfCheckGPT | CascadeGuard | Paper's claim |
|---|---|---|---|---|
| Cascading failure detection rate (recall) | PENDING | PENDING | PENDING | — |
| Reduction vs. best single-step baseline | — | — | PENDING | 34.7% |
| False alarm rate (empirical) | PENDING | PENDING | PENDING | <0.2% (target) |
| Latency overhead p50 / p95 / p99 (ms/step) | — | — | PENDING | 12 ms (p? unspecified in paper) |

Break results out **per dataset** (HaluEval-RAG / RAGTruth / AgentHallu) in addition to the aggregate — the paper reports an aggregate number, but per-dataset variance is useful information for understanding where the framework is stronger/weaker.

---

## 5. Theorem 1 empirical validation (CHAF ≤ depth+1)

- Ran on: PENDING (which dataset / how many traces)
- Violations found: PENDING (should be zero if α_ij estimation is correct — see AGENTS.md §2 invariant #2)
- CHAF vs. depth plot/summary: PENDING — attach or link

If violations are found here, this is a bug-hunting entry point, not evidence the theorem is wrong (see AGENTS.md §2).

---

## 6. Track A vs Track B ablation (α_ij estimation)

- Correlation between Track A (attention) and Track B (surrogate) on the subset where both are computable: PENDING
- Detection-quality gap when using Track B only (simulating a closed-API-only deployment): PENDING
- Conclusion / recommendation for production default: PENDING → log final decision in DECISIONS.md D-002

---

## 7. Monitor placement approximation quality

- Brute-force optimal computed on synthetic DAGs (≤12 nodes): PENDING
- Greedy result on same DAGs: PENDING
- Empirical approximation ratio observed: PENDING (compare to theoretical (1−1/e) ≈ 63% bound)

---

## 8. Honest gap analysis

This section matters more than the headline numbers. For any metric where the implementation doesn't match the paper's claim, write down the likely cause — don't just report the number.

| Metric | Paper | Reproduced | Gap | Likely cause |
|---|---|---|---|---|
| Detection improvement | 34.7% | PENDING | PENDING | PENDING |
| Latency overhead | 12ms | PENDING | PENDING | PENDING |

Common legitimate reasons a gap might exist (fill in only if actually true for your run, don't assume by default):
- Track B (surrogate α_ij) used instead of Track A due to closed-model constraints — expected to underperform the paper's presumably white-box setup.
- Smaller eval sample size than the paper's full dataset run.
- Different embedding/NLI models than whatever the paper's implementation used (paper doesn't fully specify this).

---

## 9. Summary (fill in last, once everything above is real)

One paragraph: does the implementation substantiate the paper's core claims, where does it fall short, and what's the most defensible one-sentence summary of the result if asked in an interview.
