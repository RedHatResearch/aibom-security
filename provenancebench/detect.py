"""Agent-editable detector. pbench-eval loads this file from the bench root.

Must not read ground-truth labels — eval passes only ref/sus, revisions, and stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Pred = Literal["related", "unrelated", "abstain"]

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "hub"


@dataclass(frozen=True)
class Score:
    pred: Pred
    confidence: float
    note: str = ""


def score(
    ref: str,
    sus: str,
    *,
    ref_rev: str | None = None,
    sus_rev: str | None = None,
    stack: str,
) -> Score:
    """Score one pair. Replace this body with real weight-level signals."""
    _ = (ref, sus, ref_rev, sus_rev, stack, CACHE_DIR)

    # Baseline stub: always abstain. Real detectors must use weights, not labels.
    return Score(pred="abstain", confidence=0.0, note="stub abstain")
