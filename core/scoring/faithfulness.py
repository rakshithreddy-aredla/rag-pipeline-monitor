"""Faithfulness Scorer — implements Eq. 3 (F(s_i)) and Eq. 4 (H(s_i) = 1 − F(s_i)).

F(s_i) is cosine similarity between output embedding and retrieved-context
embedding r_i (README §4.2). Long retrieval sets are chunk-AVERAGED,
never silently truncated.
"""

from __future__ import annotations

import numpy as np

from core.embeddings import EmbeddingFn
from core.graph.pipeline_graph import Document, cosine_similarity

DEFAULT_MAX_CONCAT_CHARS = 6000


def retrieval_concatenation(
    docs: list[Document], max_concat_chars: int = DEFAULT_MAX_CONCAT_CHARS
) -> str:
    """r_i = concatenation of R(s_i) (Eq. 3 input), capped by length."""
    joined = "\n".join(d.content for d in docs)
    return joined[:max_concat_chars]


def chunk_average_embedding(
    docs: list[Document],
    embed_fn: EmbeddingFn,
    max_concat_chars: int = DEFAULT_MAX_CONCAT_CHARS,
) -> np.ndarray | None:
    """Embed a long R(s_i) by averaging per-chunk embeddings, not truncating.

    Returns None when there are no documents (caller decides the semantics —
    a retrieve-less step's faithfulness is undefined, not zero).
    """
    texts = [d.content for d in docs if d.content.strip()]
    if not texts:
        return None
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and size + len(text) > max_concat_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        chunks.append("\n".join(current))
    return np.mean([embed_fn(c) for c in chunks], axis=0)


def faithfulness(
    o_i: str,
    retrieved_docs: list[Document],
    embed_fn: EmbeddingFn,
    max_concat_chars: int = DEFAULT_MAX_CONCAT_CHARS,
) -> float | None:
    """F(s_i) — Eq. 3. Cosine similarity between output and context embedding.

    Returns None when the step has no retrieved context to be faithful *to*
    (e.g. a plan step). Callers treat None as "not applicable", never as 0.
    """
    context_vec = chunk_average_embedding(retrieved_docs, embed_fn, max_concat_chars)
    if context_vec is None:
        return None
    return cosine_similarity(embed_fn(o_i), context_vec)


def hallucination_score(
    o_i: str,
    retrieved_docs: list[Document],
    embed_fn: EmbeddingFn,
    max_concat_chars: int = DEFAULT_MAX_CONCAT_CHARS,
) -> float | None:
    """H(s_i) = 1 − F(s_i) — Eq. 4, in [0, 1]. None ⇔ no retrieval context."""
    f = faithfulness(o_i, retrieved_docs, embed_fn, max_concat_chars)
    return None if f is None else float(np.clip(1.0 - f, 0.0, 1.0))
