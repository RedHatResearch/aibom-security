"""Test-run dependency rules: predicates, skip reasons, and YAML/JSON loading.

YAML surface uses short names (``upstream``, ``status``, ``score``, ``gt``);
``load_rules`` maps those into :class:`Requirement` fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, assert_never, cast, get_args

import yaml

from aibom_verifier.types import TestOutcome, TestStatus

RequirementKind = Literal["status", "score"]
_VALID_TEST_STATUSES = frozenset(get_args(TestStatus))

_DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "data" / "t1_default.yaml"


@dataclass(frozen=True, slots=True)
class Requirement:
    """One upstream gate that must hold before a downstream test may run."""

    upstream_test_id: str
    kind: RequirementKind
    expect_status: TestStatus | None = None
    score_key: str | None = None
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class Rule:
    """One test in a test run, plus the requirements that gate it."""

    test_id: str
    requires: list[Requirement] = field(default_factory=list)


def default_rules_path() -> Path:
    """Path to the packaged default T1 rule list."""
    return _DEFAULT_RULES_PATH


def _parse_requirement(raw: Any) -> Requirement:
    if not isinstance(raw, dict):
        raise ValueError(f"requirement must be a mapping: {raw!r}")
    upstream = raw.get("upstream")
    if not isinstance(upstream, str) or not upstream:
        raise ValueError(f"requirement missing string 'upstream': {raw!r}")

    if "status" in raw:
        status = raw["status"]
        if not isinstance(status, str):
            raise ValueError(f"requirement 'status' must be a string: {raw!r}")
        if status not in _VALID_TEST_STATUSES:
            allowed = ", ".join(sorted(_VALID_TEST_STATUSES))
            raise ValueError(f"requirement 'status' must be one of {allowed}: {raw!r}")
        return Requirement(
            upstream_test_id=upstream,
            kind="status",
            expect_status=cast(TestStatus, status),
        )

    if "score" in raw or "gt" in raw:
        score_key = raw.get("score")
        threshold = raw.get("gt")
        if not isinstance(score_key, str) or score_key == "":
            raise ValueError(f"score requirement needs string 'score': {raw!r}")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(f"score requirement needs numeric 'gt': {raw!r}")
        return Requirement(
            upstream_test_id=upstream,
            kind="score",
            score_key=score_key,
            threshold=float(threshold),
        )

    raise ValueError(
        f"requirement needs 'status' or 'score'+'gt': {raw!r}",
    )


def _parse_rule(raw: Any) -> Rule:
    if not isinstance(raw, dict):
        raise ValueError(f"rule must be a mapping: {raw!r}")
    test_id = raw.get("test_id")
    if not isinstance(test_id, str) or not test_id:
        raise ValueError(f"rule missing string 'test_id': {raw!r}")
    requires_raw = raw.get("requires") or []
    if not isinstance(requires_raw, list):
        raise ValueError(f"rule 'requires' must be a list: {raw!r}")
    return Rule(
        test_id=test_id,
        requires=[_parse_requirement(item) for item in requires_raw],
    )


def _load_rules_from_path(rules_path: Path) -> list[Rule]:
    text = rules_path.read_text(encoding="utf-8")
    suffix = rules_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported rules file suffix: {rules_path}")

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(f"rules file must contain a top-level 'rules' list: {rules_path}")
    rules_raw = data["rules"]
    if not isinstance(rules_raw, list):
        raise ValueError(f"'rules' must be a list: {rules_path}")
    return [_parse_rule(item) for item in rules_raw]


@lru_cache(maxsize=1)
def _cached_default_rules() -> tuple[Rule, ...]:
    return tuple(_load_rules_from_path(default_rules_path()))


def load_rules(path: Path | None = None) -> list[Rule]:
    """Load a rule list from YAML or JSON.

    Short YAML names map to :class:`Requirement` fields:
    ``upstream`` → ``upstream_test_id``; ``status`` → kind/status;
    ``score`` + ``gt`` → ``score_key`` / ``threshold``.
    """
    if path is None:
        # Copy so callers cannot mutate the cached default objects.
        return [
            Rule(test_id=rule.test_id, requires=list(rule.requires))
            for rule in _cached_default_rules()
        ]
    return _load_rules_from_path(path)


def requirement_satisfied(
    req: Requirement,
    outcomes: dict[str, TestOutcome],
) -> bool:
    """Return whether ``req`` holds given outcomes already produced.

    Missing upstream outcomes fail closed (not satisfied).
    Score gates use a strict greater-than comparison against ``threshold``.
    """
    upstream = outcomes.get(req.upstream_test_id)
    if upstream is None:
        return False

    if req.kind == "status":
        if req.expect_status is None:
            return False
        return upstream.status == req.expect_status

    if req.kind == "score":
        if req.score_key is None or req.threshold is None:
            return False
        score = upstream.scores.get(req.score_key)
        if score is None:
            return False
        return score > req.threshold

    assert_never(req.kind)


def skip_reason_for(req: Requirement, upstream: TestOutcome) -> str:
    """Generic skip reason when a requirement is not satisfied.

    Live T1 gates prefer the hybrid helpers in ``verdict_synthesize``; this
    covers score gates and other cases without a dedicated helper.
    """
    if req.kind == "status":
        return upstream.status

    if req.kind == "score":
        return "score_below_threshold"

    assert_never(req.kind)
