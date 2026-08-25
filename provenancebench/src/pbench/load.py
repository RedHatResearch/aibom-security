"""Bench data: read YAML and resolve model pins."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml

BenchRoot = Path(__file__).resolve().parents[2]
ModelSide = Literal["ref", "sus"]
LabelConfidence = Literal["verified", "inferred", "disputed"]
VALID_LABEL_CONFIDENCE = frozenset({"verified", "inferred", "disputed"})


def bench_root() -> Path:
    return BenchRoot


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"expected mapping in {path}"
        raise ValueError(msg)
    return data


def load_kinds(root: Path | None = None) -> list[dict[str, Any]]:
    data = _read_yaml((root or bench_root()) / "kinds.yaml")
    kinds = data.get("kinds")
    if not isinstance(kinds, list):
        msg = "kinds.yaml: kinds must be a list"
        raise ValueError(msg)
    return kinds


def load_cases(root: Path | None = None) -> list[dict[str, Any]]:
    data = _read_yaml((root or bench_root()) / "pairs.yaml")
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        msg = "pairs.yaml: pairs must be a list"
        raise ValueError(msg)
    return pairs


def _parse_smoke_text(text: str) -> list[str]:
    ids: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            ids.append(stripped)
    return ids


def load_smoke_ids(root: Path | None = None) -> list[str]:
    ids, _ = load_smoke(root)
    return ids


def load_smoke(root: Path | None = None) -> tuple[list[str], str]:
    """Return smoke case ids and a short content hash for SUMMARY fingerprints."""
    path = (root or bench_root()) / "smoke.txt"
    raw = path.read_bytes()
    ids = _parse_smoke_text(raw.decode("utf-8"))
    smoke_sha = hashlib.sha256(raw).hexdigest()[:12]
    return ids, smoke_sha


def file_sha256(path: Path, *, short: bool = True) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12] if short else digest


def manifest_hashes(root: Path | None = None) -> dict[str, str]:
    root = root or bench_root()
    _, smoke_sha = load_smoke(root)
    return {
        "pairs_sha256": file_sha256(root / "pairs.yaml"),
        "kinds_sha256": file_sha256(root / "kinds.yaml"),
        "smoke_sha256": smoke_sha,
    }


def label_confidence(case: dict[str, Any]) -> LabelConfidence:
    value = case.get("label_confidence", "inferred")
    if value not in VALID_LABEL_CONFIDENCE:
        msg = f"case {case.get('id')}: label_confidence must be verified, inferred, or disputed"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def model_pin(case: dict[str, Any], side: ModelSide) -> tuple[str, str | None]:
    """Return (repo_id, revision_sha_or_none) for ref or sus."""
    repo_id = case[side]
    if not isinstance(repo_id, str):
        msg = f"case {case.get('id')}: {side} must be a string repo id"
        raise ValueError(msg)
    rev_key = f"{side}_rev"
    rev = case.get(rev_key)
    if rev is None:
        return repo_id, None
    if not isinstance(rev, str) or not rev:
        msg = f"case {case.get('id')}: {rev_key} must be a non-empty string"
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


def kind_index(kinds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in kinds:
        kind = row.get("kind")
        if not isinstance(kind, str):
            msg = "each kind row needs a string kind"
            raise ValueError(msg)
        out[kind] = row
    return out
