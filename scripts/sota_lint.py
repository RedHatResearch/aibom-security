#!/usr/bin/env python3
"""Validate docs/sota/catalog.yaml against the schema rules and repo invariants.

Local-first: run before every commit that touches the catalog.
  uv run --group docs python scripts/sota_lint.py
Exits 0 on pass; prints actionable file-level errors otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "docs/sota/catalog.yaml"
FIXTURES = ROOT / "smokes/fixtures.yaml"
PB_PAIRS = ROOT / "provenancebench/pairs.yaml"

CLASSES = {"static", "inference", "moe", "engine", "tooling", "field-case"}
ACCESS = {"white", "grey", "black", "mixed", "na"}
STATUS = {"new", "evaluated", "reference", "parked", "rejected"}
VERDICTS = {"adopt", "benchmark", "reject", "cite"}
CONFIDENCE = {"verified", "inferred", "lead"}
PRIORITY = {"next", "soon", "later", "none"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FORBIDDEN_IN_NOTES = re.compile(r"[\|`]")
MD_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
NAME_IS_LINK = re.compile(r"^\[[^\]]+\]\([^)]+\)$")
EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
NAME_MAX = 120

errors: list[str] = []


def err(entry: dict | None, msg: str) -> None:
    tag = f"catalog.yaml [{entry.get('id')}]: " if entry else "catalog.yaml: "
    errors.append(tag + msg)


def main() -> int:
    if not CATALOG.exists():
        print("catalog.yaml: FATAL — file not found", file=sys.stderr)
        return 2
    raw = CATALOG.read_text()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        print(f"catalog.yaml: FATAL — YAML parse error: {e}", file=sys.stderr)
        return 2

    entries = data.get("entries") or []
    if data.get("schema_version") != 1:
        err(None, "schema_version must be 1")

    fixture_ids: set[str] = set()
    for registry in (FIXTURES, PB_PAIRS):
        if not registry.exists():
            err(None, f"fixture registry not found: {registry}")
            continue
        fx = yaml.safe_load(registry.read_text()) or {}
        pairs = fx.get("pairs") or []
        if isinstance(pairs, dict):  # smokes/fixtures.yaml: mapping of fixtures
            pairs = list(pairs.values())
        for pair in pairs:  # provenancebench/pairs.yaml: list with id: keys
            if isinstance(pair, dict):
                fixture_ids.add(str(pair.get("id", pair.get("ref", ""))))
        for key in fx.get("deferred") or {}:
            fixture_ids.add(f"deferred.{key}")

    seen_ids: set[str] = set()
    seen_sources: dict[str, str] = {}
    smokes_dir = ROOT / "smokes"
    tracked_smokes = (
        {p.name for p in smokes_dir.iterdir() if p.is_dir()} if smokes_dir.exists() else set()
    )

    for e in entries:
        if not isinstance(e, dict):
            err(None, f"entry is not a mapping: {e!r}")
            continue
        eid = e.get("id")
        # identity
        if not eid or not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", str(eid)):
            err(e, "id must be a kebab-case slug")
            eid = eid or "<bad>"
        if eid in seen_ids:
            err(e, "duplicate id")
        seen_ids.add(eid)
        # enums
        if e.get("class") not in CLASSES:
            err(e, f"class must be one of {sorted(CLASSES)}")
        if e.get("access") not in ACCESS:
            err(e, f"access must be one of {sorted(ACCESS)}")
        if e.get("status") not in STATUS:
            err(e, f"status must be one of {sorted(STATUS)}")
        if e.get("confidence") is None:
            e["confidence"] = "inferred"  # documented default; see docs/sota/AGENTS.md
        if e.get("priority") is None:
            e["priority"] = "none"
        if e.get("confidence") not in CONFIDENCE:
            err(e, f"confidence must be one of {sorted(CONFIDENCE)}")
        if e.get("priority") not in PRIORITY:
            err(e, f"priority must be one of {sorted(PRIORITY)}")
        # dates quoted (a bare date becomes datetime.date via PyYAML)
        for f in ("first_seen", "last_verified"):
            v = e.get(f)
            if v is not None and not (isinstance(v, str) and DATE.match(v)):
                err(e, f"{f} must be a quoted ISO date string")
        # required fields
        name = str(e.get("name") or "")
        if not name:
            err(e, "name is required")
        elif len(name) > NAME_MAX:
            err(e, f"name exceeds {NAME_MAX} chars ({len(name)})")
        if "[" in name and not NAME_IS_LINK.fullmatch(name):
            err(e, "name mixes a link with text; use either plain text or a single full link")
        if "|" in name:
            err(e, "name must not contain '|'")
        if len(MD_LINK.findall(name)) > 1:
            err(e, "name carries more than one markdown link; keep one")
        if not e.get("source"):
            err(e, "source is required")
        # verdict ⇒ findings
        v = e.get("verdict")
        if v is not None:
            if not isinstance(v, dict) or v.get("decision") not in VERDICTS:
                err(e, f"verdict.decision must be one of {sorted(VERDICTS)}")
            elif not e.get("findings"):
                err(e, "verdict requires findings (link the deep-dive comment)")
        # duplicate normalized source URLs
        src = e.get("source")
        if src:
            norm = re.sub(r"[?#].*$", "", str(src).rstrip("/"))
            if norm in seen_sources:
                err(
                    e,
                    f"duplicate source URL (also used by '{seen_sources[norm]}');"
                    " add notes or merge",
                )
            seen_sources[norm] = str(eid)
        # smoke path exists
        sm = e.get("smoke")
        if sm:
            if not (ROOT / sm).exists():
                err(e, f"smoke path does not exist: {sm}")
            if sm != "smokes/_lib/" and sm.rstrip("/").split("/")[-1] not in tracked_smokes:
                err(e, f"smoke directory not tracked: {sm}")
        # fixtures exist
        for fx in e.get("fixtures") or []:
            if fx not in fixture_ids:
                err(e, f"fixture id not in a fixture registry: {fx}")
        # notes hygiene
        notes = e.get("notes")
        if notes:
            if len(str(notes)) > 300:
                err(e, "notes exceed 300 chars")
            if FORBIDDEN_IN_NOTES.search(str(notes)):
                err(e, "notes must not contain '|' or backticks")
        # public-copy hygiene on every string field, findings included
        for f, val in e.items():
            values = val if isinstance(val, list) else [val]
            for v in values:
                if not isinstance(v, str):
                    continue
                if ".scratch/" in v or "/Users/" in v or "/home/" in v:
                    err(e, f"{f} must not reference .scratch/ or home paths")
                if EMAIL.search(v):
                    err(e, f"{f} must not contain an email address")

    if errors:
        for line in errors:
            print("ERROR " + line, file=sys.stderr)
        print(f"\n{len(errors)} error(s). Fix catalog.yaml, then re-run.", file=sys.stderr)
        return 1

    n_by_class: dict[str, int] = {}
    for e in entries:
        n_by_class[str(e.get("class"))] = n_by_class.get(str(e.get("class")), 0) + 1
    print(
        f"OK — {len(entries)} entries: "
        + ", ".join(f"{v} {k}" for k, v in sorted(n_by_class.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
