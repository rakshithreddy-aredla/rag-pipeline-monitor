    # Changelog

All notable changes to CascadeGuard are logged here, dated, terse. This is *what shipped*, not *why* (that's `DECISIONS.md`) and not *what's planned* (that's `TODO.md`).

Any agent finishing a session should add an entry here — one or two lines per meaningful change, not a full diff summary. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

Order: newest at top. Use `Added` / `Changed` / `Fixed` / `Docs` as rough categories where useful.

---

## [Unreleased]

### Added
- **v1 implementation of the full research core** (TODO Phases 1–9): PipelineGraph with Kahn topo + depth + dynamic edge inference; faithfulness scoring (Eq. 3–4) with chunk-averaged long-context handling; propagation α_ij with exact Track A Eq. 6 (verified against paper PDF; per-row normalization) and Track B NLI-proxy/counterfactual surrogates, all `alpha_method`-tagged; cascade engine H_cascade via single topological pass (Eq. 5), CHAF (Eq. 7), runtime Theorem 1 invariant check; semantic entropy (Eq. 9) with bidirectional-NLI clustering and a call-site S* budget guard; CHRS (Eq. 10); CUSUM detector (Eq. 11–12) with bisection control-limit inversion, empirical calibration, per-run state pool; greedy monitor placement (§II.H) + brute-force validation oracle; flow-based root-cause localization.
- **SDK** (Phase 10): StepRecorder + generic path; minimal LangChain callback handler and LlamaIndex query-engine wrapper (import-guarded).
- **Core Engine API** (Phase 11): FastAPI service — run/step ingestion, full analysis endpoint persisting scores/edges/CUSUM state/alerts to the repository, cached cheap-check lookup, calibration + placement-refresh endpoints.
- **Alerting** (Phase 12): pluggable sink interface, log/webhook/Slack sinks, per-run fire-once dispatcher.
- **Storage**: README §5 schema verbatim (`storage/schema.sql`) for Postgres+pgvector; in-memory repository as runnable default; docker-compose (pgvector + Redis).
- **Eval harness** (v1 scope): synthetic Fig.-1 trace generators, random-DAG Theorem 1 sweep, CUSUM ARL0 simulation, `eval/run_eval.py` CLI; loud not-faked stubs for dataset loaders and RAGAS/SelfCheckGPT baselines.
- **Tests**: hand-computed unit tests for every equation (per AGENTS.md §5), Theorem 1 violation-detection test, ARL0 simulation tests, placement approximation-ratio test, end-to-end engine + API integration tests.

### Docs
- Initial documentation set created: README.md, TODO.md, AGENTS.md, AGENT_OPERATING_PROMPT.md, DECISIONS.md, EVAL_REPORT.md, GLOSSARY.md, .env.example, CHANGELOG.md, CLAUDE.md, LICENSE.
- No implementation code written yet — project is at Phase 0 (setup) per TODO.md. *(superseded same day by v1 build above)*

---

<!--
Template for future entries:

## [Phase X complete] — YYYY-MM-DD

### Added
- Short description of what was implemented.

### Changed
- Anything that changed from the original plan and why (link to DECISIONS.md entry if it's a real decision).

### Fixed
- Bugs found and fixed, especially anything touching the Theorem 1 invariant or CUSUM calibration.
-->
