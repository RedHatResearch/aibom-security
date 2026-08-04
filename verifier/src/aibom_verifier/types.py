"""Core result types shared across the verification pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

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

# Alias used by the internal gate chain (same taxonomy as the public Verdict).
FinalVerdict = Verdict

TestStatus = Literal["pass", "fail", "skip", "error"]
Compatibility = Literal["compatible", "incompatible", "unsupported", "insufficient_evidence"]
BaseSource = Literal["card", "cli"]

POLICY_VERSION = "t1-1"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The JSON envelope every ``aibom verify`` invocation returns."""

    target: str
    base: str | None
    verdict: Verdict
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ModelRef:
    repo_id: str
    revision: str
    sha: str

    def to_dict(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "revision": self.revision, "sha": self.sha}


@dataclass
class TestOutcome:
    __test__ = False

    test_id: str
    status: TestStatus
    compatibility: Compatibility | None = None
    scores: dict[str, float] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    skipped_because: dict[str, str] | None = None  # {"upstream", "reason"} only
    artifacts: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestOutcome:
        """Rebuild from :meth:`to_dict` / remote ``run_test`` JSON stdout."""
        return cls(
            test_id=data["test_id"],
            status=data["status"],
            compatibility=data.get("compatibility"),
            scores=data.get("scores", {}),
            reason_codes=data.get("reason_codes", []),
            skipped_because=data.get("skipped_because"),
            artifacts=data.get("artifacts", []),
            detail=data.get("detail", {}),
        )


def outcome_from_remote_dict(payload: object, *, context: str) -> TestOutcome:
    """Parse a remote ``TestOutcome`` dict; raise ``RuntimeError`` if invalid."""
    if not isinstance(payload, dict) or "test_id" not in payload or "status" not in payload:
        raise RuntimeError(f"{context} is not a TestOutcome object")
    return TestOutcome.from_dict(cast(dict[str, Any], payload))


@dataclass
class RunResult:
    target: ModelRef
    base: ModelRef
    base_source: BaseSource
    support_class: str
    tests: list[TestOutcome]
    final_verdict: FinalVerdict
    cache: dict[str, list[str]]  # {"hits": [...], "misses": [...]}
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "base": {**self.base.to_dict(), "source": self.base_source},
            "support_class": self.support_class,
            "tests": [t.to_dict() for t in self.tests],
            "final_verdict": self.final_verdict,
            "cache": self.cache,
            "policy_version": self.policy_version,
        }


class CompareStartError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)
