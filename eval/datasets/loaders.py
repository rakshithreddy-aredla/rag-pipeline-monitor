"""Dataset loaders (TODO Phase 14) - v1 skeletons.

Each loader converts a benchmark dataset into PipelineGraph traces with
ground-truth labels. They REQUIRE the datasets on disk and raise with download
instructions rather than pretending to run - per AGENTS.md, stubs must be loud,
never silently faked.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_NOTE = (
    "Download {name} and place it at {{path}}; see README section 6 step 1. "
    "Loader implementation lands with TODO Phase 14 once the raw format is in hand."
)


class DatasetNotAvailable(FileNotFoundError):
    pass


def load_halu_eval_rag(path: Path):
    raise DatasetNotAvailable(REQUIRED_NOTE.format(name="HaluEval-RAG", path=path))


def load_ragtruth(path: Path):
    raise DatasetNotAvailable(REQUIRED_NOTE.format(name="RAGTruth", path=path))


def load_agenthallu(path: Path):
    raise DatasetNotAvailable(REQUIRED_NOTE.format(name="AgentHallu", path=path))
