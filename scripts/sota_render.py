#!/usr/bin/env python3
"""Render docs/sota/README.md and the wiki snapshot from docs/sota/catalog.yaml.

  uv run --group docs python scripts/sota_render.py                # write README.md
  uv run --group docs python scripts/sota_render.py --check        # CI drift check
  uv run --group docs python scripts/sota_render.py --wiki PATH    # write the wiki page

README output is deterministic: stable sort, no timestamps, do-not-edit banner.
The wiki page carries the publishing commit and date, so it is written only when
publishing, never checked in CI.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "docs/sota/catalog.yaml"
OUT_README = ROOT / "docs/sota/README.md"

REPO_URL = "https://github.com/RedHatResearch/aibom-security"
HUB = f"{REPO_URL}/issues/26"
STALE_BEFORE = "2026-05-01"

BANNER = "<!-- GENERATED from docs/sota/catalog.yaml by scripts/sota_render.py — DO NOT EDIT. -->"
CLASS_ORDER = ["static", "inference", "moe", "engine", "tooling", "field-case"]
CLASS_TITLES = {
    "static": "White-box static (primary)",
    "inference": "Needs-inference (secondary)",
    "moe": "MoE routing / expert signals",
    "engine": "Engines / benchmarks",
    "tooling": "Inspection and quant tooling",
    "field-case": "Field cases",
}

# Catalog names may already carry a markdown link; wrapping those would nest brackets.
NAME_IS_LINK = re.compile(r"^\[.+\]\(.+\)$")


def esc(s: str) -> str:
    return str(s).replace("|", "\\|")


def verdict_cell(e: dict) -> str:
    v = e.get("verdict")
    if not v:
        return "—"
    return f"{v['decision']} ({v['date']})" if v.get("date") else str(v.get("decision"))


def name_cell(e: dict, fallback_url: str) -> str:
    name = str(e.get("name", ""))
    if NAME_IS_LINK.match(name):
        return name
    return f"[{esc(name)}]({fallback_url})" if fallback_url else esc(name)


def smoke_cell(e: dict, base: Path, wiki: bool) -> str:
    smoke = e.get("smoke")
    if not smoke:
        return "—"
    if wiki:
        href = f"{REPO_URL}/tree/main/{str(smoke).strip('/')}/"
    else:
        # README links are relative to docs/sota/; the wiki needs an absolute URL.
        href = os.path.relpath(ROOT / str(smoke), base).replace(os.sep, "/") + "/"
    return f"[`{smoke}`]({href})"


def row(e: dict, base: Path, wiki: bool) -> str:
    link = name_cell(e, str(e.get("source", "")))
    smoke = smoke_cell(e, base, wiki)
    n_findings = len(e.get("findings") or [])
    return f"| {link} | {esc(e.get('status', ''))} | {verdict_cell(e)} | {n_findings} | {smoke} |"


def wiki_header(n_entries: int) -> list[str]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()[:7]
    day = datetime.now(UTC).date().isoformat()
    return [
        f"Source: docs/sota/catalog.yaml @ {sha} · {day} · {n_entries} entries.",
        f"Generated; do not edit — open an issue. Scope and conventions: [issue #26]({HUB}).",
        "",
    ]


def render(entries: list[dict], title: str, wiki: bool = False) -> str:
    by_class: dict[str, list[dict]] = {c: [] for c in CLASS_ORDER}
    for e in entries:
        by_class.setdefault(str(e.get("class")), []).append(e)

    out = [f"# {title}", ""]
    if wiki:
        out += wiki_header(len(entries))
    else:
        out = [BANNER, *out]
        total = len(entries)
        decided = sum(1 for e in entries if e.get("verdict"))
        out += [
            f"**{total} entries · {decided} with verdicts** — see the [hub issue]({HUB}) "
            "for scope and conventions.",
            "",
        ]

    # Next up
    nxt = [e for e in entries if e.get("priority") in ("next", "soon")]
    if nxt:
        out += ["## Next up", ""]
        out += ["| Entry | Priority | Status |", "|---|---|---|"]
        for e in sorted(nxt, key=lambda x: ({"next": 0, "soon": 1}[x["priority"]], str(x["id"]))):
            out.append(f"| {name_cell(e, HUB)} | {e['priority']} | {esc(e.get('status', ''))} |")
        out.append("")

    for c in CLASS_ORDER:
        rows = sorted(by_class.get(c, []), key=lambda x: str(x.get("id")))
        if not rows:
            continue
        out += [f"## {CLASS_TITLES[c]} ({len(rows)})", ""]
        out += ["| Entry | Status | Verdict | Deep dives | Smoke |", "|---|---|---|---|---|"]
        for e in rows:
            out.append(row(e, OUT_README.parent, wiki))
        out.append("")

    stale = [e for e in entries if e.get("last_verified") and e["last_verified"] < STALE_BEFORE]
    if stale:
        out += [
            f"## Stale (not verified since {STALE_BEFORE})",
            "",
            ", ".join(f"`{e['id']}`" for e in stale),
            "",
        ]

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    ap.add_argument("--wiki", metavar="PATH", help="also write the wiki page to PATH")
    args = ap.parse_args(argv)
    if args.check and args.wiki:
        ap.error("--check and --wiki cannot be combined")

    if not CATALOG.exists():
        print("catalog.yaml not found", file=sys.stderr)
        return 2
    data = yaml.safe_load(CATALOG.read_text())
    entries = data.get("entries") or []

    title = "SOTA catalog — provenance fingerprints"
    readme = render(entries, title)
    if args.check:
        current = OUT_README.read_text() if OUT_README.exists() else ""
        if current != readme:
            print(
                "docs/sota/README.md is stale — run scripts/sota_render.py and commit",
                file=sys.stderr,
            )
            return 1
        print("OK — README.md up to date")
        return 0

    OUT_README.write_text(readme)
    try:
        where = OUT_README.relative_to(ROOT)
    except ValueError:
        where = OUT_README
    print(f"wrote {where} ({len(entries)} entries)")
    if args.wiki:
        page = Path(args.wiki)
        page.write_text(render(entries, title, wiki=True))
        print(f"wrote {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
