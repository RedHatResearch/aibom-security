"""Core result types shared across the verification pipeline.

The actual fingerprinting/comparison logic (architecture-hash gate, block-0
shape and byte comparison) lands with Milestone 1's FR issues — see
https://github.com/RedHatResearch/aibom-security/milestone/1. This module
only defines the stable output shape so the CLI has something real to return
in the meantime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Verdict = Literal[
    "verified_derivative",
    "fraudulent_claim",
    "incompatible",
    "unsupported",
    "insufficient_evidence",
]
"""Only ``verified_derivative`` may later feed an AI-BOM lineage entry.

Every other verdict is a refusal to confirm, not a soft failure — the
verifier defaults to abstaining (``unsupported`` / ``insufficient_evidence``)
rather than over-confirming on architectures or evidence it cannot honestly
assess.
"""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The JSON envelope every ``aibom verify`` invocation returns."""

    target: str
    base: str | None
    verdict: Verdict
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
