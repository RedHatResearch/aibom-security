"""Bench data: read the v1 manifest and resolve immutable model pins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml

BenchRoot = Path(__file__).resolve().parents[2]
ModelSide = Literal["subject", "reference"]
VALID_EXPECTED = frozenset({"related", "unrelated", "unsupported"})
VALID_STACK = frozenset({"dense", "moe"})
VALID_CONFIDENCE = frozenset({"verified", "inferred", "disputed"})
VALID_CHALLENGE_TYPES = frozenset(
    {"unrelated_reference", "misattributed_parent", "non_comparable_input"}
)
VALID_LINEAGE_SCOPE = frozenset({"direct", "transitive"})
REQUIRED_CASE_FIELDS = frozenset(
    {
        "id",
        "stack",
        "kind",
        "subject",
        "subject_rev",
        "reference",
        "reference_rev",
        "expected",
        "evidence",
    }
)


def bench_root() -> Path:
    return BenchRoot


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"expected mapping in {path}"
        raise ValueError(msg)
    return data


def load_kinds(root: Path | None = None) -> list[str]:
    data = _read_yaml((root or bench_root()) / "kinds.yaml")
    if data.get("schema_version") != 1:
        msg = "kinds.yaml: schema_version must be 1"
        raise ValueError(msg)
    kinds = data.get("kinds")
    if not isinstance(kinds, list):
        msg = "kinds.yaml: kinds must be a list"
        raise ValueError(msg)
    for index, kind in enumerate(kinds, start=1):
        _validate_kind(kind, index)
    kind_index(kinds)
    return kinds


def load_cases(root: Path | None = None) -> list[dict[str, Any]]:
    data = _read_yaml((root or bench_root()) / "pairs.yaml")
    if data.get("schema_version") != 1:
        msg = "pairs.yaml: schema_version must be 1"
        raise ValueError(msg)
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        msg = "pairs.yaml: pairs must be a list"
        raise ValueError(msg)
    for case in pairs:
        _validate_case(case)
    case_by_id(pairs)
    kinds = kind_index(load_kinds(root))
    for case in pairs:
        if case["kind"] not in kinds:
            msg = f"{case['id']}: unknown kind {case['kind']}"
            raise ValueError(msg)
    return pairs


def load_cohorts(root: Path | None = None) -> dict[str, dict[str, Any]]:
    data = _read_yaml((root or bench_root()) / "cohorts.yaml")
    if data.get("schema_version") != 1:
        msg = "cohorts.yaml: schema_version must be 1"
        raise ValueError(msg)
    cohorts = data.get("cohorts")
    if not isinstance(cohorts, dict):
        msg = "cohorts.yaml: cohorts must be a mapping"
        raise ValueError(msg)
    if not all(isinstance(name, str) and isinstance(row, dict) for name, row in cohorts.items()):
        msg = "cohorts.yaml: every cohort must be a named mapping"
        raise ValueError(msg)
    for name, cohort in cohorts.items():
        _validate_cohort(name, cohort)
    return cohorts


def load_cohort(name: str, root: Path | None = None) -> dict[str, Any]:
    cohort = load_cohorts(root).get(name)
    if cohort is None:
        msg = f"unknown cohort: {name}"
        raise ValueError(msg)
    ids = cohort["ids"]
    cases = case_by_id(load_cases(root))
    for case_id in ids:
        case = cases.get(case_id)
        if case is None:
            msg = f"cohort {name}: unknown case id {case_id}"
            raise ValueError(msg)
    return cohort


def manifest_hashes(root: Path | None = None) -> dict[str, str]:
    root = root or bench_root()
    manifest = b"\0".join((root / name).read_bytes() for name in ("pairs.yaml", "kinds.yaml"))
    return {
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_case(case: object) -> None:
    if not isinstance(case, dict):
        msg = "each case must be a mapping"
        raise ValueError(msg)
    case = cast(dict[str, Any], case)
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        msg = "each case needs a non-empty string id"
        raise ValueError(msg)
    missing = REQUIRED_CASE_FIELDS - case.keys()
    if missing:
        msg = f"{case_id}: missing required field {sorted(missing)[0]}"
        raise ValueError(msg)
    if case["stack"] not in VALID_STACK:
        msg = f"{case_id}: stack must be dense or moe"
        raise ValueError(msg)
    if not isinstance(case["kind"], str) or not case["kind"]:
        msg = f"{case_id}: kind must be a non-empty string"
        raise ValueError(msg)
    for side in ("subject", "reference"):
        if not isinstance(case[side], str) or not case[side]:
            msg = f"{case_id}: {side} must be a non-empty string"
            raise ValueError(msg)
        revision = case[f"{side}_rev"]
        if not isinstance(revision, str) or len(revision) != 40 or not _is_sha(revision):
            msg = f"{case_id}: {side}_rev must be a full immutable SHA"
            raise ValueError(msg)
    if case["expected"] not in VALID_EXPECTED:
        msg = f"{case_id}: expected must be related, unrelated, or unsupported"
        raise ValueError(msg)
    evidence = case["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"confidence", "sources", "note"}:
        msg = f"{case_id}: evidence must contain confidence, sources, and note"
        raise ValueError(msg)
    evidence = cast(dict[str, Any], evidence)
    if evidence["confidence"] not in VALID_CONFIDENCE:
        msg = f"{case_id}: evidence.confidence is invalid"
        raise ValueError(msg)
    if (
        not isinstance(evidence["sources"], list)
        or not evidence["sources"]
        or not all(isinstance(source, str) and source for source in evidence["sources"])
    ):
        msg = f"{case_id}: evidence.sources must be a non-empty list of strings"
        raise ValueError(msg)
    if not isinstance(evidence["note"], str) or not evidence["note"]:
        msg = f"{case_id}: evidence.note must be a non-empty string"
        raise ValueError(msg)
    if "context" in case and (not isinstance(case["context"], str) or not case["context"]):
        msg = f"{case_id}: context must be a non-empty string when present"
        raise ValueError(msg)
    lineage_scope = case.get("lineage_scope")
    if case["expected"] == "related":
        if lineage_scope not in VALID_LINEAGE_SCOPE:
            msg = f"{case_id}: related cases need lineage_scope direct or transitive"
            raise ValueError(msg)
        if "challenge" in case:
            msg = f"{case_id}: related cases must not define challenge"
            raise ValueError(msg)
    elif lineage_scope is not None:
        msg = f"{case_id}: only related cases may define lineage_scope"
        raise ValueError(msg)
    challenge = case.get("challenge")
    if challenge is not None:
        if not isinstance(challenge, dict) or not set(challenge) <= {
            "type",
            "difficulty",
            "confounders",
        }:
            msg = f"{case_id}: challenge has unsupported fields"
            raise ValueError(msg)
        if challenge.get("type") not in VALID_CHALLENGE_TYPES:
            msg = f"{case_id}: challenge.type is invalid"
            raise ValueError(msg)
        if "difficulty" in challenge and (
            not isinstance(challenge["difficulty"], str) or not challenge["difficulty"]
        ):
            msg = f"{case_id}: challenge.difficulty must be a non-empty string"
            raise ValueError(msg)
        if "confounders" in challenge and (
            not isinstance(challenge["confounders"], list)
            or not all(
                isinstance(confounder, str) and confounder
                for confounder in challenge["confounders"]
            )
        ):
            msg = f"{case_id}: challenge.confounders must be a list of non-empty strings"
            raise ValueError(msg)


def _validate_kind(row: object, index: int) -> None:
    if not isinstance(row, str) or not row:
        msg = f"kinds.yaml: row {index} must be a non-empty string"
        raise ValueError(msg)


def _validate_cohort(name: str, cohort: dict[str, Any]) -> None:
    ids = cohort.get("ids")
    if not isinstance(ids, list) or not all(
        isinstance(case_id, str) and case_id for case_id in ids
    ):
        msg = f"cohort {name}: ids must be a list of non-empty strings"
        raise ValueError(msg)
    if not ids:
        msg = f"cohort {name}: ids must not be empty"
        raise ValueError(msg)
    if len(ids) != len(set(ids)):
        msg = f"cohort {name}: ids must not contain duplicates"
        raise ValueError(msg)


def _is_sha(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


def model_pin(case: dict[str, Any], side: ModelSide) -> tuple[str, str]:
    """Return a subject or reference repository and its immutable revision."""
    repo_id = case[side]
    if not isinstance(repo_id, str):
        msg = f"case {case.get('id')}: {side} must be a string repo id"
        raise ValueError(msg)
    rev_key = f"{side}_rev"
    rev = case.get(rev_key)
    if not isinstance(rev, str) or len(rev) != 40 or not _is_sha(rev):
        msg = f"case {case.get('id')}: {rev_key} must be a full immutable SHA"
        raise ValueError(msg)
    return repo_id, rev


def case_by_id(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str):
            msg = "each case needs a string id"
            raise ValueError(msg)
        if case_id in out:
            msg = f"duplicate case id: {case_id}"
            raise ValueError(msg)
        out[case_id] = case
    return out


def kind_index(kinds: list[str]) -> set[str]:
    out: set[str] = set()
    for kind in kinds:
        if kind in out:
            msg = f"duplicate kind: {kind}"
            raise ValueError(msg)
        out.add(kind)
    return out
