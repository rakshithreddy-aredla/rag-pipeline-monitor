"""Embedding backends behind one callable interface.

`core/` takes an `EmbeddingFn` everywhere — the concrete backend is a config
decision (configs/default.yaml `embeddings.backend`), still open per DECISIONS.md
D-003. The hashing backend is deterministic and dependency-free so the whole
system runs and tests green with zero model downloads; swap to
sentence-transformers for real workloads.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class EmbeddingFn(Protocol):
    def __call__(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag-of-character-trigrams embedder (dependency-free).

    NOT semantically deep — lexical overlap only. Exists so the pipeline is
    runnable and testable without downloading models; production should use
    the sentence-transformers backend (README §7).
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = "".join(c.lower() if c.isalnum() else " " for c in text)
        for i in range(len(normalized) - 2):
            trigram = normalized[i : i + 3]
            idx = int.from_bytes(hashlib.md5(trigram.encode()).digest()[:4], "big") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "bge-base-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "embeddings.backend='sentence-transformers' requires "
                "`pip install cascadeguard[embeddings]` (DECISIONS.md D-003)"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode(text), dtype=np.float32)


def load_embedder(config: dict) -> EmbeddingFn:
    """Build the configured embedder from configs/default.yaml `embeddings:` block."""
    backend = config.get("backend", "hashing")
    if backend == "hashing":
        return HashingEmbedder(dim=int(config.get("dim", 256)))
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(str(config.get("model", "bge-base-en-v1.5")))
    raise ValueError(f"unknown embeddings backend {backend!r}")


__all__ = ["EmbeddingFn", "HashingEmbedder", "SentenceTransformerEmbedder", "load_embedder"]
