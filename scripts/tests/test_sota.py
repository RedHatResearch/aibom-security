"""Tests for scripts/sota_lint.py and scripts/sota_render.py (happy path + failure modes)."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
LINT = ROOT / "scripts/sota_lint.py"
RENDER = ROOT / "scripts/sota_render.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, cwd=ROOT
    )


GOOD_ENTRY = {
    "id": "test-entry",
    "name": "Test Entry",
    "class": "static",
    "access": "white",
    "status": "new",
    "source": "https://arxiv.org/abs/0000.00000",
    "first_seen": "2026-10-30",
    "confidence": "verified",
    "priority": "none",
}


@pytest.fixture()
def catalog(tmp_path, monkeypatch):
    def write(entries: list[dict]) -> Path:
        cat = tmp_path / "catalog.yaml"
        cat.write_text(yaml.safe_dump({"schema_version": 1, "entries": entries}, sort_keys=False))
        return cat

    lint = _load("sota_lint_under_test", LINT)
    monkeypatch.setattr(lint, "CATALOG", tmp_path / "catalog.yaml")
    monkeypatch.setattr(lint, "FIXTURES", ROOT / "smokes/fixtures.yaml")
    write([dict(GOOD_ENTRY)])
    return write, lint


def test_lint_happy_path(catalog):
    _, lint = catalog
    assert lint.main() == 0


def test_lint_bad_enum(catalog):
    write, lint = catalog
    write([dict(GOOD_ENTRY, status="candidate")])
    assert lint.main() == 1


def test_lint_duplicate_id(catalog):
    write, lint = catalog
    write([dict(GOOD_ENTRY), dict(GOOD_ENTRY)])
    assert lint.main() == 1


def test_lint_verdict_requires_findings(catalog):
    write, lint = catalog
    write([dict(GOOD_ENTRY, verdict={"decision": "adopt", "date": "2026-10-30"})])
    assert lint.main() == 1


def test_lint_duplicate_source_url(catalog):
    write, lint = catalog
    write([dict(GOOD_ENTRY), dict(GOOD_ENTRY, id="other-entry", name="Other")])
    assert lint.main() == 1


def test_lint_name_too_long(catalog):
    write, lint = catalog
    write([dict(GOOD_ENTRY, name="x" * 121)])
    assert lint.main() == 1


def test_lint_name_with_two_links(catalog):
    write, lint = catalog
    name = "[A](https://a.example) and [B](https://b.example)"
    write([dict(GOOD_ENTRY, name=name)])
    assert lint.main() == 1


def test_lint_name_with_one_link_ok(catalog):
    write, lint = catalog
    write([dict(GOOD_ENTRY, name="[A](https://a.example)")])
    assert lint.main() == 0


def test_lint_bare_date_rejected(catalog):
    _, lint = catalog
    lint.CATALOG.write_text(
        "schema_version: 1\n"
        "entries:\n"
        "  - id: bare-date\n"
        "    name: Bare\n"
        "    class: static\n"
        "    access: white\n"
        "    status: new\n"
        "    source: https://example.com\n"
        "    confidence: verified\n"
        "    priority: none\n"
        "    first_seen: 2026-10-30\n"  # bare YAML date parses as datetime.date
    )
    assert lint.main() == 1


def test_lint_malformed_yaml(tmp_path, monkeypatch):
    lint = _load("sota_lint_malformed", LINT)
    cat = tmp_path / "catalog.yaml"
    cat.write_text("entries: [oops\n")
    monkeypatch.setattr(lint, "CATALOG", cat)
    assert lint.main() == 2


def test_lint_missing_file(tmp_path, monkeypatch):
    lint = _load("sota_lint_missing", LINT)
    monkeypatch.setattr(lint, "CATALOG", tmp_path / "nope.yaml")
    assert lint.main() == 2


def test_render_check_detects_drift(tmp_path, monkeypatch):
    render = _load("sota_render_under_test", RENDER)
    monkeypatch.setattr(render, "ROOT", tmp_path)
    cat = tmp_path / "catalog.yaml"
    cat.write_text(yaml.safe_dump({"schema_version": 1, "entries": [GOOD_ENTRY]}, sort_keys=False))
    out = tmp_path / "README.md"
    monkeypatch.setattr(render, "CATALOG", cat)
    monkeypatch.setattr(render, "OUT_README", out)
    assert render.main([]) == 0
    assert render.main(["--check"]) == 0
    out.write_text("hand-edited\n")
    assert render.main(["--check"]) == 1


def test_render_wiki_requires_path():
    render = _load("sota_render_wiki_arg", RENDER)
    with pytest.raises(SystemExit) as exc:
        render.main(["--wiki"])
    assert exc.value.code == 2


def test_render_wiki_page_has_publish_header_and_absolute_urls(tmp_path, monkeypatch):
    render = _load("sota_render_wiki", RENDER)
    entries = [
        dict(GOOD_ENTRY, smoke="smokes/tensorlock/"),
        dict(GOOD_ENTRY, id="other-entry", name="Other", source="https://example.com/other"),
    ]
    cat = tmp_path / "catalog.yaml"
    cat.write_text(yaml.safe_dump({"schema_version": 1, "entries": entries}, sort_keys=False))
    monkeypatch.setattr(render, "CATALOG", cat)
    monkeypatch.setattr(render, "OUT_README", tmp_path / "README.md")
    page = tmp_path / "wiki" / "SOTA-Catalog.md"
    page.parent.mkdir()

    assert render.main(["--wiki", str(page)]) == 0
    text = page.read_text()
    assert re.search(
        r"^Source: docs/sota/catalog\.yaml @ [0-9a-f]{7} · \d{4}-\d{2}-\d{2} · 2 entries\.$",
        text,
        re.MULTILINE,
    )
    assert "Generated; do not edit — open an issue." in text
    assert "https://github.com/RedHatResearch/aibom-security/tree/main/smokes/tensorlock/" in text
    assert render.BANNER not in text
    assert not (ROOT / "docs/sota/wiki-draft.md").exists()


def test_render_keeps_existing_name_link_and_fixes_smoke_path(tmp_path, monkeypatch):
    render = _load("sota_render_links", RENDER)
    entry = dict(
        GOOD_ENTRY,
        name="[Linked Name](https://example.com/source)",
        source="https://example.com/fallback",
        smoke="smokes/tensorlock/",
    )
    cat = tmp_path / "catalog.yaml"
    cat.write_text(yaml.safe_dump({"schema_version": 1, "entries": [entry]}, sort_keys=False))
    docs = tmp_path / "docs" / "sota"
    docs.mkdir(parents=True)
    monkeypatch.setattr(render, "ROOT", tmp_path)
    monkeypatch.setattr(render, "CATALOG", cat)
    monkeypatch.setattr(render, "OUT_README", docs / "README.md")

    assert render.main([]) == 0
    text = (docs / "README.md").read_text()
    assert "| [Linked Name](https://example.com/source) |" in text
    assert "[[Linked Name]" not in text
    assert "[`smokes/tensorlock/`](../../smokes/tensorlock/)" in text


def test_render_stale_heading_names_the_cutoff(tmp_path, monkeypatch):
    render = _load("sota_render_stale", RENDER)
    entry = dict(GOOD_ENTRY, last_verified="2025-12-01")
    cat = tmp_path / "catalog.yaml"
    cat.write_text(yaml.safe_dump({"schema_version": 1, "entries": [entry]}, sort_keys=False))
    monkeypatch.setattr(render, "CATALOG", cat)
    monkeypatch.setattr(render, "OUT_README", tmp_path / "README.md")

    assert render.main([]) == 0
    text = (tmp_path / "README.md").read_text()
    assert "## Stale (not verified since 2026-05-01)" in text
    assert "not verified in 180+ days" not in text
    assert "`test-entry`" in text


def test_render_escapes_pipe():
    render = _load("sota_render_esc", RENDER)
    assert render.esc("a|b") == "a\\|b"


def test_real_catalog_lints_and_renders():
    """The committed catalog is the primary fixture: lint + render + drift check must pass."""
    lint = _run(LINT)
    assert lint.returncode == 0, lint.stderr
    render = _run(RENDER, "--check")
    assert render.returncode == 0, render.stderr
