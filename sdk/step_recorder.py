"""StepRecorder — in-process SDK instrumentation (TODO Phase 10).

Non-blocking capture of agent steps into a PipelineGraph. The generic path is
a plain recorder API; framework adapters translate callback events into
`record_step` calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from core.graph.pipeline_graph import (
    Document,
    PipelineGraph,
    Step,
    build_dynamic_edges,
)


class StepRecorder:
    """Collects Steps for ONE pipeline run and finalizes the DAG."""

    def __init__(self, template_id: str = "adhoc") -> None:
        self.graph = PipelineGraph(template_id)
        self._start: float = time.time()

    def record_step(
        self,
        step_id: str,
        kind: str,
        output: str = "",
        retrieved_docs: list[Document] | None = None,
        predecessors: list[str] | None = None,
        tool_invocation: dict[str, Any] | None = None,
        attention_weights: Any | None = None,
    ) -> Step:
        step = Step(
            id=step_id,
            kind=kind,  # type: ignore[arg-type]
            output=output,
            retrieved_docs=retrieved_docs or [],
            predecessors=predecessors or [],
            tool_invocation=tool_invocation,
            attention_weights=attention_weights,
            timestamp=time.time() - self._start,
        )
        self.graph.add_step(step)
        return step

    def finalize(
        self, infer_edges_with_embeddings: Callable[[str], np.ndarray] | None = None
    ) -> PipelineGraph:
        """Close the run: run dynamic edge inference over undeclared data flow.

        `infer_edges_with_embeddings` is an EmbeddingFn; without it only n-gram
        overlap inference runs (README section 4.1 two-stage heuristic).
        """
        if infer_edges_with_embeddings is not None:
            build_dynamic_edges(self.graph, embed_fn=infer_edges_with_embeddings)
        else:
            build_dynamic_edges(self.graph)
        return self.graph
