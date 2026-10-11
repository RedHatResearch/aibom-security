"""Agent-editable detector. pbench-eval loads this file from the bench root.

Must not read curated labels — eval passes only subject/reference ids, revisions, and stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["related", "unrelated", "abstain"]


@dataclass(frozen=True)
class Result:
    verdict: Verdict
    score: float | None
    diagnostics: dict[str, object]
    note: str = ""
    abstain_reason: str | None = None


def score(
    subject: str,
    reference: str,
    *,
    subject_rev: str | None = None,
    reference_rev: str | None = None,
    stack: str,
) -> Result:
    """Score one pair. Replace this body with real weight-level signals."""
    # Baseline stub: always abstain. Real detectors must use weights, not labels.
    return Result(
        verdict="abstain",
        score=None,
        diagnostics={},
        note="stub abstain",
        abstain_reason="method_not_implemented",
    )
