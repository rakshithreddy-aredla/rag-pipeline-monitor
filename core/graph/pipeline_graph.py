"""Pipeline Graph Builder — implements §4.1 / paper Def. 1 and Eq. 1.

G = (V, E): steps are nodes; an edge (s_i, s_j) exists iff s_j's context c_j
includes s_i's output o_i (Eq. 1: c_i = R(s_i) ∪ {o_j | (s_j, s_i) ∈ E}).
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

StepKind = Literal["plan", "retrieve", "reason", "tool_call", "synthesize"]
VALID_KINDS: tuple[str, ...] = ("plan", "retrieve", "reason", "tool_call", "synthesize")

NL = chr(10)  # newline constant (keeps source free of embedded escapes)


class CycleError(ValueError):
    """Raised when the step graph contains a cycle — Eq. 5 needs a DAG evaluation order."""


@dataclass
class Document:
    """One retrieved document in R(s_i)."""

    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Step:
    """A single pipeline step s_i (paper Def. 1)."""

    id: str
    kind: StepKind
    output: str  # o_i (Eq. 2)
    retrieved_docs: list[Document] = field(default_factory=list)  # R(s_i)
    predecessors: list[str] = field(default_factory=list)  # incoming edges {(s_j, s_i)}
    tool_invocation: dict | None = None
    attention_weights: np.ndarray | None = None  # A^(j), Track A only
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"invalid step kind {self.kind!r}; expected one of {VALID_KINDS}")

    def context(self, resolved_outputs: dict[str, str]) -> str:
        """Assemble c_i per Eq. 1: R(s_i) ∪ {o_j | (s_j, s_i) ∈ E}.

        `resolved_outputs` maps predecessor step id -> its output string.
        Retrieved docs come first, then predecessor outputs in edge order.
        """
        parts = [d.content for d in self.retrieved_docs]
        parts += [resolved_outputs[p] for p in self.predecessors if p in resolved_outputs]
        return "\n".join(parts)


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def ngram_overlap(source: str, target: str, n: int = 3) -> float:
    """Fraction of `source`'s word n-grams that appear in `target`.

    Directional: measures how much of the earlier step's output reappears in the
    later step's context — the signal that an edge (source -> target) exists.
    """
    src_tokens = _tokenize(source)
    tgt_tokens = _tokenize(target)
    if len(src_tokens) < n:
        # Too short for n-grams: fall back to unigram containment.
        if not src_tokens:
            return 0.0
        tgt_set = set(tgt_tokens)
        hits = sum(1 for t in src_tokens if t in tgt_set)
        return hits / len(src_tokens)
    src_grams = {" ".join(src_tokens[i : i + n]) for i in range(len(src_tokens) - n + 1)}
    if not src_grams:
        return 0.0
    tgt_grams = {" ".join(tgt_tokens[i : i + n]) for i in range(len(tgt_tokens) - n + 1)}
    return len(src_grams & tgt_grams) / len(src_grams)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def edge_exists(
    source_output: str,
    target_context: str,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    ngram_threshold: float = 0.15,
    similarity_threshold: float = 0.62,
    ngram_size: int = 3,
) -> bool:
    """Decide whether edge (s_i, s_j) exists: does o_i feed c_j? (README §4.1).

    Two-stage heuristic for dynamic agents:
      1. directional n-gram overlap >= `ngram_threshold`  (verbatim/near-verbatim reuse)
      2. else embedding cosine(o_i, c_j) >= `similarity_threshold` (paraphrased reuse),
         only when an embedder is available.
    Static-template pipelines bypass this entirely — their edges are exact.
    """
    if not source_output.strip():
        return False
    if ngram_overlap(source_output, target_context, ngram_size) >= ngram_threshold:
        return True
    if embed_fn is None:
        return False
    return cosine_similarity(embed_fn(source_output), embed_fn(target_context)) >= (
        similarity_threshold
    )


class PipelineGraph:
    """Incremental DAG of pipeline steps (paper Def. 1)."""

    def __init__(self, pipeline_template_id: str = "adhoc") -> None:
        self.pipeline_template_id = pipeline_template_id
        self.nodes: dict[str, Step] = {}
        self._edges_out: dict[str, list[str]] = defaultdict(list)
        self._edges_in: dict[str, list[str]] = defaultdict(list)
        self._topo_cache: list[str] | None = None
        self._depth_cache: dict[str, int] = {}

    # -- construction -----------------------------------------------------

    def add_step(self, step: Step) -> None:
        if step.id in self.nodes:
            raise ValueError(f"duplicate step id {step.id!r}")
        for pred in step.predecessors:
            if pred not in self.nodes:
                raise ValueError(
                    f"step {step.id!r} declares predecessor {pred!r} which is not yet added"
                )
        self.nodes[step.id] = step
        for pred in step.predecessors:
            self._add_edge_raw(pred, step.id)

    def add_edge(self, src_id: str, dst_id: str) -> None:
        """Add a data-flow edge (used by static templates and dynamic inference).

        Does NOT check for cycles — a templated DAG is assumed acyclic by its
        author, and dynamic inference only ever connects earlier→later steps.
        A genuine cycle surfaces loudly at `topo_order()` (CycleError).
        """
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise KeyError(f"cannot add edge ({src_id!r}, {dst_id!r}): node missing")
        if dst_id in self._edges_out.get(src_id, []):
            return  # idempotent
        self._add_edge_raw(src_id, dst_id)

    def _add_edge_raw(self, src_id: str, dst_id: str) -> None:
        self._edges_out[src_id].append(dst_id)
        self._edges_in[dst_id].append(src_id)
        self._topo_cache = None
        self._depth_cache.clear()

    # -- traversal --------------------------------------------------------

    def edges_into(self, node_id: str) -> list[str]:
        return list(self._edges_in.get(node_id, []))

    def edges_out_of(self, node_id: str) -> list[str]:
        return list(self._edges_out.get(node_id, []))

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(src, dst) for src, dsts in self._edges_out.items() for dst in dsts]

    def sources(self) -> list[str]:
        return [nid for nid in self.nodes if not self._edges_in.get(nid)]

    def sinks(self) -> list[str]:
        return [nid for nid in self.nodes if not self._edges_out.get(nid)]

    def topo_order(self) -> list[str]:
        """Kahn's algorithm — required for O(|V|+|E|) H_cascade evaluation (§II.C).

        Raises CycleError on a cyclic graph: Eq. 5's evaluation order depends on
        acyclicity, so a violation must be loud, never silently tolerated.
        """
        if self._topo_cache is not None:
            return list(self._topo_cache)
        indegree = {nid: len(self._edges_in.get(nid, [])) for nid in self.nodes}
        queue: deque[str] = deque(sorted(nid for nid, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in sorted(self._edges_out.get(nid, [])):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.nodes):
            raise CycleError("pipeline graph contains a cycle; H_cascade requires a DAG")
        self._topo_cache = order
        return list(order)

    def depth(self, node_id: str) -> int:
        """Longest path (in edges) from any source to `node_id` — needed for Thm. 1."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        if node_id in self._depth_cache:
            return self._depth_cache[node_id]
        preds = self._edges_in.get(node_id, [])
        value = 0 if not preds else 1 + max(self.depth(p) for p in preds)
        self._depth_cache[node_id] = value
        return value


def build_dynamic_edges(
    graph: PipelineGraph,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    ngram_threshold: float = 0.15,
    similarity_threshold: float = 0.62,
    ngram_size: int = 3,
) -> None:
    """Infer missing edges for free-form agent loops (README §4.1 dynamic path).

    For every ordered pair (earlier, later), if `later`'s context (its retrieved
    docs + currently-known predecessor outputs) contains evidence of `earlier`'s
    output, add the edge. Heuristic by design — see README §10 on mislabeled edges.
    Mutates `graph` in place.
    """
    # Capture (insertion) order IS causal order for dynamically recorded runs.
    # Topological order would be wrong here: before inference the graph may be
    # edge-less, and Kahn tie-breaking is alphabetical, not temporal.
    order = list(graph.nodes.keys())
    for j_pos, later_id in enumerate(order):
        later = graph.nodes[later_id]
        # Observable content of the later step: its retrieved docs plus its own
        # output. Undeclared predecessor outputs aren't available as raw context
        # in free-form loops - paraphrase reuse shows up in o_j itself.
        observed = NL.join([d.content for d in later.retrieved_docs] + [later.output])
        known = {pred: graph.nodes[pred].output for pred in later.predecessors}
        context_so_far = later.context(known)
        for earlier_id in order[:j_pos]:
            if earlier_id in later.predecessors:
                continue
            if edge_exists(
                graph.nodes[earlier_id].output,
                observed or context_so_far,
                embed_fn=embed_fn,
                ngram_threshold=ngram_threshold,
                similarity_threshold=similarity_threshold,
                ngram_size=ngram_size,
            ):
                later.predecessors.append(earlier_id)
                graph.add_edge(earlier_id, later_id)
