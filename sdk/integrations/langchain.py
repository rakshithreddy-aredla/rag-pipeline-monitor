"""LangChain adapter — translates callback events into StepRecorder calls.

Import-guarded: requires langchain-core at the host app.
"""

from __future__ import annotations

from core.graph.pipeline_graph import Document


def build_langchain_handler(recorder):
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as exc:
        raise ImportError(
            "sdk.integrations.langchain requires pip install cascadeguard[langchain]"
        ) from exc

    class _CascadeGuardHandler(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, *, run_id=None, parent_run_id=None, **kwargs):
            pass

        def on_chain_end(self, outputs, *, run_id=None, parent_run_id=None, **kwargs):
            pass

        def on_retriever_end(self, serialized, results, **kwargs) -> None:
            recorder.record_step(
                step_id=str(len(recorder.graph.nodes)),
                kind="retrieve",
                output="",
                retrieved_docs=[
                    Document(doc_id=str(i), content=str(getattr(d, "page_content", d)))
                    for i, d in enumerate(results or [])
                ],
            )

        def on_llm_end(self, response, **kwargs) -> None:
            text = ""
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                text = getattr(generations[0][0], "text", "")
            recorder.record_step(
                step_id=str(len(recorder.graph.nodes)),
                kind="reason",
                output=text,
            )

    return _CascadeGuardHandler()
