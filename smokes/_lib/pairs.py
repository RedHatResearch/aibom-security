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
