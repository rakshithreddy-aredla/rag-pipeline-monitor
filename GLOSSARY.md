# GLOSSARY.md

Maps every symbol/term from the CascadeGuard paper to its code identifier and location. When you (or any agent) forget "wait, what's CHAF again" mid-debugging, this should answer it in under 10 seconds instead of requiring a re-read of README §4 or the paper itself.

---

## Core notation

| Symbol | Name | Meaning (one line) | Equation | Code identifier | Location |
|---|---|---|---|---|---|
| `P = (G, L, R)` | Agentic RAG pipeline | The full pipeline as graph + label fn + retrieval fn | Def. 1 | `PipelineGraph` | `core/graph/pipeline_graph.py` |
| `G = (V, E)` | Pipeline DAG | Steps as nodes, data-flow as edges | Def. 1 | `PipelineGraph.nodes/.edges` | `core/graph/pipeline_graph.py` |
| `s_i` | A pipeline step | One node — plan/retrieve/reason/tool_call/synthesize | Def. 1 | `Step` | `core/graph/pipeline_graph.py` |
| `c_i` | Context for step i | Retrieved docs + predecessor outputs | Eq. 1 | built from `Step.retrieved_docs` + predecessor `Step.output` | `core/graph/pipeline_graph.py` |
| `o_i` | Output of step i | `LLM(c_i, t_i; θ)` | Eq. 2 | `Step.output` | `core/graph/pipeline_graph.py` |
| `F(s_i)` | Faithfulness score | Cosine sim between output and retrieved context embeddings | Eq. 3 | `faithfulness()` | `core/scoring/faithfulness.py` |
| `H(s_i)` | Single-step hallucination score | `1 - F(s_i)` | Eq. 4 | `hallucination_score()` | `core/scoring/faithfulness.py` |
| `α_ij` | Propagation coefficient | How much of j's output attends to i's output | Eq. 6 | `propagation_coefficient_attention()` (Track A) / surrogate fns (Track B) | `core/scoring/propagation.py` |
| `A^(j)` | Cross-attention matrix of step j | Raw attention weights, needed for exact `α_ij` | — | `Step.attention_weights` | `core/graph/pipeline_graph.py` |
| `H_cascade(s_j)` | Cascading hallucination score | H(s_j) plus propagated risk from predecessors | Eq. 5 | `compute_h_cascade()` | `core/scoring/cascade_score.py` |
| `CHAF(s_j)` | Cascading Hallucination Amplification Factor | `H_cascade(s_j) / H(s_j)` — how much worse cascade risk is vs. single-step | Eq. 7 | `compute_chaf()` | `core/scoring/cascade_score.py` |
| `depth(s_j)` | Node depth | Longest path from any source to s_j | (Thm. 1) | `PipelineGraph.depth()` | `core/graph/pipeline_graph.py` |
| `SE(s_i)` | Semantic entropy | Entropy over K-sampled semantic-equivalence clusters | Eq. 9 | `semantic_entropy()` | `core/scoring/semantic_entropy.py` |
| `C_i` | Semantic equivalence classes | Clusters of K samples grouped by meaning | Eq. 9 | `cluster_by_meaning()` | `core/scoring/semantic_entropy.py` |
| `CHRS(s_i)` | Cascading Hallucination Risk Score | `λ·H_cascade + (1-λ)·SE` — the combined scalar risk | Eq. 10 | `chrs()` | `core/scoring/risk_combiner.py` |
| `λ` | Risk combiner weight | Tuning knob between structural and uncertainty risk | Eq. 10 | `config.risk_combiner.lambda` | `configs/default.yaml` — see DECISIONS.md D-001 |
| `S_n` | CUSUM statistic | Running sequential-detection score | Eq. 11 | `CusumDetector.S` | `core/detection/cusum.py` |
| `κ` (kappa) | CUSUM slack | `μ0 + δ/2` | Eq. 11 | `CusumDetector.kappa` | `core/detection/cusum.py` |
| `h` | CUSUM control limit | Alert threshold — `S_n > h` fires an alert | Eq. 11/12 | `CusumDetector.h` | `core/detection/cusum.py` |
| `ARL0` | Average Run Length under null | Expected steps between false alarms | Eq. 12 | `arl0_target` config, used in `_solve_control_limit()` | `core/detection/cusum.py`, `configs/default.yaml` |
| `μ0` | Baseline CHRS mean | Estimated from clean calibration traces | Eq. 12 | `CusumDetector.mu0`, set via `calibrate()` | `core/detection/cusum.py` |
| `σ` | Baseline CHRS stdev | Estimated from clean calibration traces | Eq. 12 | `CusumDetector.sigma`, set via `calibrate()` | `core/detection/cusum.py` |
| `δ` (delta) | Minimum detectable shift | Config knob for CUSUM sensitivity | Eq. 11 | `config.cusum.delta` | `configs/default.yaml` |
| `S*` | Optimal monitor placement | Budget-constrained subset of steps to monitor at full fidelity | Eq. 13 | `greedy_monitor_placement()` return value | `core/placement/greedy_optimizer.py` |
| `B` | Compute budget | Total budget for monitor placement | Eq. 13 | `config.monitor_placement.budget_B` | `configs/default.yaml` |
| `w_i` | Per-step monitoring cost | Cost weight used in placement optimization | Eq. 13 | `weights[s]` param | `core/placement/greedy_optimizer.py` |

---

## Key theorems (referenced, not re-derived, in code/tests)

| Name | Statement (short) | Where it's tested |
|---|---|---|
| **Theorem 1** — CHAF Depth Bound | `CHAF(s_j) ≤ depth(s_j) + 1` for all nodes | `tests/test_cascade_score.py` — must hold with zero violations, see AGENTS.md §2 |
| **Theorem 2** — NP-Hardness of monitor placement | Optimal `S*` (Eq. 13) is NP-hard via reduction from Minimum Weighted Vertex Cut | Not directly "tested" — justifies why `core/placement/` uses the greedy `(1-1/e)` approximation instead of exact optimization; validated empirically in `tests/test_placement.py` against brute-force on small synthetic DAGs |
| **Proposition 1** — False Alarm Rate Control | `ARL0` formula (Eq. 12) bounds false alarms to <0.2% at `ARL0=500` | `tests/test_cusum.py` — simulation-based validation, see TODO Phase 7 |

---

## Track A / Track B quick reference (α_ij estimation)

| | Track A (attention) | Track B (surrogate) |
|---|---|---|
| Fidelity to paper | Exact (Eq. 6) | Approximate |
| Requires | White-box model, `output_attentions=True` | Any model (API-compatible) |
| Sub-methods | — | NLI-entailment, counterfactual perturbation |
| Cost | Cheap (uses existing forward pass) | NLI: cheap. Counterfactual: expensive (extra LLM call) |
| `alpha_method` tag value | `"attention"` | `"nli_proxy"` or `"counterfactual"` |
| Code | `propagation_coefficient_attention()` | Track B functions, `core/scoring/propagation.py` |

---

## Pipeline step kinds

| Kind | Meaning |
|---|---|
| `plan` | Agent decides what to do next |
| `retrieve` | Retrieval step — populates `R(s_i)` |
| `reason` | Intermediate reasoning/synthesis step |
| `tool_call` | Agent invokes an external tool |
| `synthesize` | Final answer generation step |

---

## Fidelity levels (monitoring)

| Level | Meaning | Used for |
|---|---|---|
| `cheap` | Fast embedding-based `H(s_i)` only, no semantic entropy | Steps not in `S*` |
| `full` | Full scoring including `SE(s_i)` and (if `expensive_mode`) counterfactual `α_ij` | Steps selected by `greedy_monitor_placement()` |
