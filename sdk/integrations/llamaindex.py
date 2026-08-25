"""LlamaIndex adapter (minimal v1): records each query-engine call as a step."""

from __future__ import annotations

from core.graph.pipeline_graph import Document


def instrument_query_engine(query_engine, recorder):
    """Wrap query() so each call becomes one recorded synthesize step with its
    source nodes as retrieved docs."""
    original_query = query_engine.query

    def wrapped_query(query_str, *args, **kwargs):
        response = original_query(query_str, *args, **kwargs)
        source_nodes = getattr(response, "source_nodes", None) or []
        docs = [
            Document(doc_id=str(getattr(n, "id_", i)), content=str(getattr(n, "text", "")))
            for i, n in enumerate(source_nodes)
        ]
        recorder.record_step(
            step_id=f"li_{id(response)}",
            kind="synthesize",
            output=str(getattr(response, "response", response)),
            retrieved_docs=docs,
        )
        return response

    query_engine.query = wrapped_query
    return query_engine
