"""Default pair list from smokes/fixtures.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "fixtures.yaml"


def default_pairs() -> list[tuple[str, str, str]]:
    with open(FIXTURES_PATH) as f:
        raw = yaml.safe_load(f)
    out: list[tuple[str, str, str]] = []
    for label, spec in raw["pairs"].items():
        out.append((label, spec["ref"], spec["sus"]))
    return out


def all_pairs() -> dict[str, tuple[str, str]]:
    """Label -> (ref, sus) over both the active and deferred sections."""

    with open(FIXTURES_PATH) as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, tuple[str, str]] = {}
    for section in ("pairs", "deferred"):
        for label, spec in (raw.get(section) or {}).items():
            if isinstance(spec, dict) and spec.get("ref") and spec.get("sus"):
                out[str(label)] = (str(spec["ref"]), str(spec["sus"]))
    return out
