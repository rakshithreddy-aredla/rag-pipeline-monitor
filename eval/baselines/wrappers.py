"""Baseline detector wrappers (TODO Phase 14).

v1: loud import-guarded wrappers, NOT reimplementations - RAGAS/SelfCheckGPT
come from their own libraries so comparisons are honest.
"""

from __future__ import annotations


class BaselineUnavailable(ImportError):
    pass


def ragas_baseline():
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise BaselineUnavailable(
            "pip install ragas to run the RAGAS baseline comparison (Phase 14)"
        ) from exc
    return ragas


def selfcheckgpt_baseline():
    try:
        import selfcheckgpt  # noqa: F401
    except ImportError as exc:
        raise BaselineUnavailable(
            "SelfCheckGPT has no maintained pip package; wire the reference "
            "implementation here during Phase 14"
        ) from exc
    return selfcheckgpt
